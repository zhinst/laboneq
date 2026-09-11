# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any, Literal

from laboneq.controller.callback_protocol import MessageType
from laboneq.controller.constants import CALLBACK_CHANNEL_ABSENCE_TIMEOUT_S
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.utilities.async_helpers import wait_for

if TYPE_CHECKING:
    from collections.abc import Sequence

_logger = logging.getLogger(__name__)

DoneStatus = Literal["completed", "failed", "cancelled"]


class CallbackChannelBusy(LabOneQControllerException):
    """A live transport is already attached to the channel."""


class ClientCallbackError(LabOneQControllerException):
    """The client reported a failure while executing a callback."""


@dataclass
class CallbackReply:
    actions: list[dict[str, Any]]
    return_value: Any = None


class CallbackChannel:
    """Async bridge between the near-time run and the client's WebSocket.

    The channel lives in the controller service. The client connects over a
    WebSocket, and the route binds the service-side end of that connection with
    :meth:`attach`, releasing it with :meth:`detach` once the connection closes.
    Both are bookkeeping inside the service; nothing about them goes on the wire
    and the client never sees an attach or detach message.

    The near-time engine is synchronous, so at most one callback is outstanding
    at a time. The channel holds on to that message and to the terminal message,
    so a client that reconnects can pick up where it left off: attaching sends
    the terminal message again, or the outstanding callback if the run is still
    going. Sending a callback again never runs it twice, because the client
    recognises the ``call_id`` and answers with the reply it already computed.
    Replies carrying any other ``call_id`` are dropped here.
    """

    def __init__(
        self,
        results_url: str,
        absence_timeout_s: float = CALLBACK_CHANNEL_ABSENCE_TIMEOUT_S,
    ) -> None:
        self._results_url = results_url
        self._absence_timeout_s = absence_timeout_s
        # Both edges are tracked separately so invoke() can wait on either;
        # the channel starts life detached.
        self._attached = asyncio.Event()
        self._detached = asyncio.Event()
        self._detached.set()
        # Wall-clock of the last detach; anchors the absence window.
        self._detached_at: float | None = None
        # The live transport's outbound queue; None while detached.
        self._outbound: asyncio.Queue[dict[str, Any]] | None = None
        self._call_counter = 0
        # The single outstanding callback and its awaiter (near-time is serial).
        self._pending_message: dict[str, Any] | None = None
        self._reply_future: asyncio.Future[CallbackReply] | None = None
        # Sticky terminal message, replayed to a late/reconnecting transport.
        self._terminal_message: dict[str, Any] | None = None

    @property
    def absence_timeout_s(self) -> float:
        return self._absence_timeout_s

    @property
    def is_attached(self) -> bool:
        return self._attached.is_set()

    # ------------------------------------------------------------------
    # Transport side (WebSocket route)
    # ------------------------------------------------------------------

    def attach(self) -> asyncio.Queue[dict[str, Any]]:
        """Bind the service-side end of a newly connected client WebSocket.

        Returns the queue of messages to be sent to the client. Re-seeds it
        with the terminal message or the outstanding callback so a
        reconnecting client can resume.

        Raises:
            CallbackChannelBusy: A live transport is already attached.
        """
        if self._attached.is_set():
            raise CallbackChannelBusy(
                "A callback channel is already attached for this experiment."
            )
        self._detached_at = None
        self._outbound = asyncio.Queue()
        if self._terminal_message is not None:
            # A finished run supersedes the outstanding callback: the client
            # must not execute a callback whose reply nobody will consume.
            self._outbound.put_nowait(self._terminal_message)
        elif self._pending_message is not None:
            self._outbound.put_nowait(self._pending_message)
        self._detached.clear()
        self._attached.set()
        return self._outbound

    def detach(self) -> None:
        """Unbind the transport; the absence window starts now."""
        if not self._attached.is_set():
            return
        self._outbound = None
        self._detached_at = monotonic()
        self._attached.clear()
        self._detached.set()

    def handle_inbound(self, message: Any) -> None:
        """Process one client message; stale or unexpected ones are dropped."""
        reply_future = self._reply_future
        pending = self._pending_message
        if reply_future is None or reply_future.done() or pending is None:
            _logger.warning("Ignoring unexpected client callback message: %r", message)
            return
        if not isinstance(message, dict):
            # Also covers channel.handle_inbound(None)
            self._pending_message = None
            reply_future.set_exception(
                LabOneQControllerException(
                    f"Malformed client callback reply: {message!r}"
                )
            )
            return
        call_id = message.get("call_id")
        if call_id != pending["call_id"]:
            _logger.warning(
                "Ignoring stale client callback reply for call_id=%r", call_id
            )
            return
        # Every branch below resolves the reply, so clear the pending message
        # now; attach() must not re-send an already-answered callback.
        self._pending_message = None
        message_type = message.get("type")
        if message_type == MessageType.RESULT:
            actions = message.get("actions") or []
            if not isinstance(actions, list) or not all(
                isinstance(action, dict) for action in actions
            ):
                reply_future.set_exception(
                    LabOneQControllerException(
                        f"Malformed actions in client callback reply: {actions!r}"
                    )
                )
                return
            reply_future.set_result(
                CallbackReply(actions=actions, return_value=message.get("return_value"))
            )
        elif message_type == MessageType.ERROR:
            error = message.get("error") or {}
            reply_future.set_exception(
                ClientCallbackError(
                    f"Client callback failed: {error.get('message', 'unknown error')}"
                )
            )
        else:
            reply_future.set_exception(
                LabOneQControllerException(
                    f"Unexpected client callback message type: {message_type!r}"
                )
            )

    # ------------------------------------------------------------------
    # Controller side
    # ------------------------------------------------------------------

    async def wait_attached(self, *, restart_window: bool = False) -> bool:
        """Wait for a transport within the absence timeout.

        The timeout runs from the last detach, or from this call if the channel
        was never attached. Returns False once it has expired. With
        ``restart_window`` it runs from this call regardless of an earlier
        detach, which is what a task needs when it first becomes eligible to run.
        """
        if restart_window:
            self._detached_at = None
        if self._attached.is_set():
            return True
        remaining = self._absence_timeout_s
        if self._detached_at is not None:
            elapsed = monotonic() - self._detached_at
            remaining = self._absence_timeout_s - elapsed
        if remaining <= 0:
            return False
        try:
            await wait_for(self._attached.wait(), remaining)
        except asyncio.TimeoutError:
            return False
        return True

    async def invoke(
        self,
        name: str,
        args: dict[str, Any],
        nt_step: Sequence[int],
        results: dict[str, Any] | None,
    ) -> CallbackReply:
        """Send a callback invocation and await the client's reply.

        Raises:
            LabOneQControllerException: The channel stayed detached beyond the
                absence window, or the reply was malformed.
            ClientCallbackError: The callback raised on the client.
        """
        if self._reply_future is not None:
            raise LabOneQControllerException(
                "A near-time callback is already outstanding on this channel."
            )
        self._call_counter += 1
        message = {
            "type": MessageType.CALLBACK.value,
            "call_id": f"c-{self._call_counter}",
            "name": name,
            "args": args,
            "nt_step": list(nt_step),
            "results_url": self._results_url,
            "results": results,
        }
        reply_future: asyncio.Future[CallbackReply] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_message = message
        self._reply_future = reply_future
        try:
            if self._attached.is_set():
                assert self._outbound is not None
                self._outbound.put_nowait(message)
            # Resend-and-wait across reconnects: each turn re-arms the absence
            # timeout, then blocks until the reply arrives or the transport drops.
            while not reply_future.done():
                if not await self.wait_attached():
                    raise LabOneQControllerException(
                        f"Client callback channel was absent for more than "
                        f"{self._absence_timeout_s:.0f} s while waiting for "
                        f"callback {name!r}."
                    )
                # attach() has sent the pending message again; wake on detach so
                # the absence timeout is enforced.
                detached = asyncio.create_task(self._detached.wait())
                try:
                    waiters: set[asyncio.Future[Any]] = {reply_future, detached}
                    await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                finally:
                    detached.cancel()
            return reply_future.result()
        finally:
            self._pending_message = None
            self._reply_future = None

    def notify_done(
        self, status: DoneStatus, error: dict[str, Any] | None = None
    ) -> None:
        """Store and (if attached) send the terminal message.

        A transport attaching after the run ended receives it immediately.
        """
        self._terminal_message = {
            "type": MessageType.DONE.value,
            "status": status,
            "error": error,
        }
        if self._attached.is_set():
            assert self._outbound is not None
            self._outbound.put_nowait(self._terminal_message)
