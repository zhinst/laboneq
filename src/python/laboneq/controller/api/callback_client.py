# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Client-side execution of near-time callbacks served over the callback channel.

`ClientCallbackSession` is transport-agnostic: it consumes decoded messages from
the `WS /v1/experiments/{uuid}/callbacks` socket and produces the reply messages,
including the `call_id` dedup and stored-reply replay required by the
reconnect/resume protocol.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Mapping

import numpy as np

from laboneq.controller.callback_protocol import MessageType, json_failure
from laboneq.core.exceptions import AbortExecution, LabOneQException
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.executor.executor import ExecutorBase, LoopingMode
from laboneq.serializers.implementations.numpy_array import NumpyArraySerializer

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq.controller.toolkit_adapter import ToolkitDevices
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.result.results import Results


class _ClientCallCollector(ExecutorBase):
    """Collect the near-time calls of an execution tree."""

    def __init__(self) -> None:
        super().__init__(looping_mode=LoopingMode.NEAR_TIME_ONLY | LoopingMode.ONCE)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def nt_callback_handler(
        self,
        func_name: str,
        args: dict[str, Any],
    ) -> None:
        self.calls.append((func_name, args))


def preflight_client_callbacks(
    scheduled_experiment: ScheduledExperiment,
    callbacks: Mapping[str, Callable[..., Any]],
) -> set[str]:
    """Validate the client near-time callbacks of an experiment before submission.

    Callbacks on the controller service are not checked here; only calls
    whose name is provided in *callbacks* are dispatched to the client, and
    those must have JSON-serializable arguments — complex values are not
    supported and must be decomposed into real/imaginary parts. Returns the
    names of the client-served callbacks; an empty set means the client has
    nothing to serve and skips the callback channel.
    """
    collector = _ClientCallCollector()
    collector.run(scheduled_experiment.execution)
    served: set[str] = set()
    for name, args in collector.calls:
        if name not in callbacks:
            continue
        failure = json_failure(args)
        if failure is not None:
            raise LabOneQException(
                f"Arguments of client near-time callback '{name}' must be"
                f" JSON-serializable: {failure}"
            )
        served.add(name)
    return served


class ClientRuntimeContext:
    """Runtime context handed to a near-time callback running on the client.

    Mirrors the controller-side `RuntimeContext`: the action methods record
    actions that are sent back to the controller service in the callback reply.
    """

    def __init__(self, nt_step: list[int], results: Results):
        self._nt_step = nt_step
        self._results = results
        self._actions: list[dict[str, Any]] = []

    @property
    def actions(self) -> list[dict[str, Any]]:
        """The actions recorded so far, in call order."""
        return self._actions

    @property
    def nt_step(self) -> list[int]:
        """The near-time loop indices of the current callback invocation."""
        return self._nt_step

    @property
    def results(self) -> Results:
        """The experiment results accumulated so far.

        Acquired results and near-time callback return values are fetched
        lazily from the controller service on access.
        """
        return self._results

    @property
    def devices(self) -> ToolkitDevices:
        """Not available on the client; register the callback on the service."""
        raise LabOneQException(
            "Instrument devices are not available in client near-time callbacks."
            " Register this callback on the controller service to run it next to"
            " the instruments instead."
        )

    @property
    def emulated(self) -> bool:
        """Not available on the client; register the callback on the service."""
        raise LabOneQException(
            "The emulation state is not available in client near-time callbacks."
            " Register this callback on the controller service to run it there"
            " instead."
        )

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ) -> None:
        """Replace the samples of a pulse for the next near-time steps.

        Only sampled values are supported on the client; a `Pulse` object
        cannot be transferred to the controller service.
        """
        if isinstance(pulse_or_array, Pulse):
            raise LabOneQException(
                "Replacing a pulse with a Pulse object is not supported in"
                " client near-time callbacks; pass the sampled values instead."
            )
        uid = pulse_uid if isinstance(pulse_uid, str) else pulse_uid.uid
        self._actions.append(
            {
                "action": "replace_pulse",
                "pulse_uid": uid,
                "pulse": NumpyArraySerializer.to_json_dict(np.asarray(pulse_or_array)),
            }
        )

    def replace_phase_increment(
        self, parameter_uid: str, new_value: int | float
    ) -> None:
        """Replace a phase increment value for the next near-time steps."""
        if isinstance(new_value, np.generic):
            new_value = new_value.item()
        self._actions.append(
            {
                "action": "replace_phase_increment",
                "parameter_uid": parameter_uid,
                "value": new_value,
            }
        )

    def abort_execution(self) -> None:
        """Abort the execution of the experiment after this callback."""
        # The action carries the abort to the controller, the raise stops the callback body.
        self._actions.append({"action": "abort_execution"})
        raise AbortExecution


@dataclass(frozen=True)
class Done:
    """Terminal channel message: the run ended from the client's perspective."""

    status: str
    error: dict[str, Any] | None


@dataclass(frozen=True)
class PendingCall:
    """A callback invocation decoded from the channel, ready to execute."""

    call_id: str
    name: str
    func: Callable[..., Any]
    ctx: ClientRuntimeContext
    args: dict[str, Any]


class ClientCallbackSession:
    """Transport-agnostic client side of the near-time callback channel.

    `handle_message` decodes one server message into a `Done`, a ready-to-send
    reply (duplicate or unknown-callback), or a `PendingCall` that the transport
    runs via `execute` / `execute_async`.

    A failing user callback is reported to the service as an `error` reply and
    also kept in `user_error`. Then, `run_experiment` raises the original exception,
    with its traceback, once the run has ended.
    """

    def __init__(
        self,
        callbacks: Mapping[str, Callable[..., Any]],
        results_factory: Callable[[], Results],
    ):
        self._callbacks = callbacks
        self._results_factory = results_factory
        self._last_call_id: str | None = None
        self._last_reply: dict[str, Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self.user_error: BaseException | None = None

    def close(self) -> None:
        """Release the event loop used for coroutine callbacks, if any.

        Called when the run ends; safe to call again.
        """
        if self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        assert self._loop_thread is not None
        self._loop_thread.join()
        self._loop.close()
        self._loop = None
        self._loop_thread = None

    def _run_coroutine(self, coro: Any) -> Any:
        """Run a coroutine callback on an event loop of our own, and wait.

        `asyncio.run` cannot be used here. It refuses to start when the calling
        thread already runs a loop, which is the normal state in a Jupyter
        kernel, and it would hand each invocation a new loop, so a callback
        could not keep a connection open from one near-time step to the next.
        """
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            self._loop_thread = threading.Thread(
                target=self._loop.run_forever,
                name="laboneq-neartime-callbacks",
                daemon=True,
            )
            self._loop_thread.start()
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def handle_message(self, message: Any) -> Done | PendingCall | dict[str, Any]:
        if not isinstance(message, dict):
            raise LabOneQException(
                f"Unexpected message from the callback channel: {message!r}"
            )
        message_type = message.get("type")
        if message_type == MessageType.DONE:
            self.close()
            return Done(status=message["status"], error=message.get("error"))
        if message_type != MessageType.CALLBACK:
            raise LabOneQException(
                f"Unexpected message type from the callback channel: {message_type!r}"
            )
        call_id = message["call_id"]
        if call_id == self._last_call_id and self._last_reply is not None:
            # The server re-sent the pending callback after a reconnect.
            return self._last_reply
        name = message["name"]
        func = self._callbacks.get(name)
        if func is None:
            return self._store_reply(
                call_id,
                {
                    "type": MessageType.ERROR.value,
                    "call_id": call_id,
                    "error": {
                        "message": f"Near-time callback '{name}' is registered"
                        " neither on the client nor on the controller service."
                    },
                },
            )
        ctx = ClientRuntimeContext(
            nt_step=list(message["nt_step"]),
            results=self._results_factory(),
        )
        return PendingCall(
            call_id=call_id, name=name, func=func, ctx=ctx, args=message["args"]
        )

    def execute(self, call: PendingCall) -> dict[str, Any]:
        """Run the callback synchronously; coroutine callbacks are run to completion."""
        try:
            value = call.func(call.ctx, **call.args)
            if inspect.iscoroutine(value):
                value = self._run_coroutine(value)
        except AbortExecution:
            return self._abort_reply(call)
        except Exception as exc:  # noqa: BLE001 - reported as an error reply
            return self._error_reply(call, exc)
        return self._result_reply(call, value)

    async def execute_async(self, call: PendingCall) -> dict[str, Any]:
        """Run the callback, awaiting an awaitable result.

        A synchronous callback runs in a worker thread so that a long-running
        one does not starve the event loop — the WebSocket must keep answering
        the server's heartbeat pings while the callback executes.
        """
        try:
            value = await asyncio.to_thread(call.func, call.ctx, **call.args)
            if inspect.isawaitable(value):
                value = await value
        except AbortExecution:
            return self._abort_reply(call)
        except Exception as exc:  # noqa: BLE001 - reported as an error reply
            return self._error_reply(call, exc)
        return self._result_reply(call, value)

    def _result_reply(self, call: PendingCall, value: Any) -> dict[str, Any]:
        failure = json_failure(value)
        if failure is not None:
            exc = LabOneQException(
                f"The return value of client near-time callback '{call.name}'"
                f" must be JSON-serializable: {failure}"
            )
            return self._error_reply(call, exc, message=str(exc))
        return self._store_reply(
            call.call_id,
            {
                "type": MessageType.RESULT.value,
                "call_id": call.call_id,
                "actions": call.ctx.actions,
                "return_value": value,
            },
        )

    def _abort_reply(self, call: PendingCall) -> dict[str, Any]:
        return self._store_reply(
            call.call_id,
            {
                "type": MessageType.RESULT.value,
                "call_id": call.call_id,
                "actions": call.ctx.actions,
                "return_value": None,
            },
        )

    def _error_reply(
        self, call: PendingCall, exc: BaseException, message: str | None = None
    ) -> dict[str, Any]:
        self.user_error = exc
        if message is None:
            message = f"{type(exc).__name__}: {exc}"
        return self._store_reply(
            call.call_id,
            {
                "type": MessageType.ERROR.value,
                "call_id": call.call_id,
                "error": {"message": message},
            },
        )

    def _store_reply(self, call_id: str, reply: dict[str, Any]) -> dict[str, Any]:
        # Kept until the next callback arrives: after a reconnect the server
        # re-sends the outstanding callback and expects the same reply again.
        self._last_call_id = call_id
        self._last_reply = reply
        return reply
