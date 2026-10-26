# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json as json_module
import logging
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import httpx
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    InvalidHandshake,
)

from laboneq import __version__
from laboneq.controller.api.async_controller_api import (
    AsyncControllerAPI,
)
from laboneq.controller.api.callback_client import (
    ClientCallbackSession,
    Done,
    PendingCall,
    preflight_client_callbacks,
)
from laboneq.controller.api.commons import (
    APIError,
    SubmissionHandle,
    encode_submission,
    fetch_acquired_results,
    fetch_neartime_callback_results,
    fetch_result_structure,
    raise_on_failed_run,
    readyz_not_ready_reason,
    reraise_controller_exception,
    ws_url,
)
from laboneq.controller.api.lazy_results import (
    LazyAcquiredResults,
    LazyNeartimeCallbackResults,
)
from laboneq.controller.callback_protocol import (
    CALLBACK_PROTOCOL_HEADER,
    CALLBACK_PROTOCOL_VERSION,
    json_default,
)
from laboneq.controller.constants import (
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_HEADER,
    READY_TIMEOUT_S,
)
from laboneq.controller.controller import SubmissionStatus
from laboneq.controller.utilities.exception import HardwareNotReadyError
from laboneq.dsl.result.results import Results
from laboneq.serializers import from_dict

if TYPE_CHECKING:
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup

_logger = logging.getLogger(__name__)

_TERMINAL_STATUSES = frozenset({SubmissionStatus.COMPLETED, SubmissionStatus.FAILED})
_RECONNECT_BACKOFF_INITIAL_S = 0.5
_RECONNECT_BACKOFF_MAX_S = 5.0
#: How long to keep retrying a callback channel the service reports as busy.
_CHANNEL_BUSY_TIMEOUT_S = 30.0


class AsyncRemoteController(AsyncControllerAPI):
    #: Total time to wait for the controller service to come up on connect.
    _LIVEZ_TIMEOUT_S: float = 60.0
    #: Delay between successive liveness polls.
    _LIVEZ_POLL_INTERVAL_S: float = 0.5
    #: Delay between successive readiness polls.
    _READYZ_POLL_INTERVAL_S: float = 0.5

    @staticmethod
    async def create(
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> AsyncRemoteController:
        remote_controller = AsyncRemoteController(
            remote_url=remote_url,
            ignore_version_mismatch=ignore_version_mismatch,
            _transport=_transport,
        )
        await remote_controller._connect()
        return remote_controller

    def __init__(
        self,
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        _transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._remote_url = remote_url
        self._ignore_version_mismatch = ignore_version_mismatch
        self._transport = _transport
        self._headers = {
            "X-LabOneQ-Client-Version": __version__,
            PROTOCOL_VERSION_HEADER: PROTOCOL_VERSION,
            CALLBACK_PROTOCOL_HEADER: CALLBACK_PROTOCOL_VERSION,
        }
        if ignore_version_mismatch is not None:
            self._headers["X-LabOneQ-Ignore-Version-Mismatch"] = str(
                ignore_version_mismatch
            ).lower()

    async def _connect(self):
        await self._wait_until_live()

    async def _wait_until_live(self) -> None:
        """Poll the controller service's `/livez` endpoint until the service is up.

        The service may still be starting up (connecting to hardware, or
        booting in emulation mode) when the client first tries to reach it;
        retry until `/livez` answers or `_LIVEZ_TIMEOUT_S` elapses.
        `get_not_ready_reason` reports whether the service can run experiments.
        """
        url = f"{self._remote_url}/livez"
        deadline = time.monotonic() + self._LIVEZ_TIMEOUT_S
        last_error = "no response received"
        while True:
            try:
                async with httpx.AsyncClient(transport=self._transport) as client:
                    resp = await client.get(url, headers=self._headers)
                if resp.status_code == 200:
                    return
                last_error = f"{resp.status_code}: {resp.text}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise APIError(
                    f"Controller service at {self._remote_url} did not come up "
                    f"within {self._LIVEZ_TIMEOUT_S:.0f}s: {last_error}"
                )
            await asyncio.sleep(self._LIVEZ_POLL_INTERVAL_S)

    async def aclose(self):
        pass

    async def get_default_devicesetup(self) -> DeviceSetup:
        data = await self._request_json("GET", "v1/devicesetup")
        return cast("DeviceSetup", from_dict(data["device_setup"]))

    async def get_instrument_topology(self) -> InstrumentTopology | None:
        data = await self._request_json("GET", "v1/instrumenttopology")
        if data.get("instrument_topology") is None:
            return None
        return cast("InstrumentTopology", from_dict(data["instrument_topology"]))

    async def get_not_ready_reason(self) -> str | None:
        try:
            async with httpx.AsyncClient(transport=self._transport) as client:
                resp = await client.get(
                    f"{self._remote_url}/readyz", headers=self._headers
                )
        except httpx.HTTPError as exc:
            return f"Cannot reach the controller service at {self._remote_url}: {exc}"
        return readyz_not_ready_reason(resp)

    async def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        while (reason := await self.get_not_ready_reason()) is not None:
            if time.monotonic() >= deadline:
                raise HardwareNotReadyError(
                    f"Not ready after {timeout_s:g} s. {reason}"
                )
            await asyncio.sleep(self._READYZ_POLL_INTERVAL_S)

    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        body = encode_submission(scheduled_experiment, coprocessor_payload_overrides)
        if handle is None:
            handle = SubmissionHandle()
        await self._request(
            "PUT",
            f"v1/experiments/{handle.hex}",
            json=body,
        )
        return handle

    async def wait_for_experiment(self, handle: SubmissionHandle) -> None:
        poll_interval = 0.5
        while True:
            exp_status = await self.get_experiment_status(handle)
            if exp_status in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(poll_interval)

    async def run_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        callbacks: dict[str, Callable[..., Any]] | None = None,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> Results:
        """Submit an experiment, serve its client near-time callbacks, and fetch results.

        Callbacks not provided by the controller service itself are dispatched
        to the matching entry in *callbacks*; their arguments are checked for
        JSON-serializability before submission. Experiments with no client-served
        callbacks fall back to plain submit-and-poll.

        A completed run stays on the service until the caller closes the
        submission. A run that fails after submission is released right away,
        before its error propagates.
        """
        callbacks = dict(callbacks or {})
        client_callbacks = preflight_client_callbacks(scheduled_experiment, callbacks)
        handle = await self.submit_experiment(
            scheduled_experiment, handle, coprocessor_payload_overrides
        )
        try:
            if client_callbacks:
                return await self._serve_run(handle, callbacks)
            return await self._poll_run(handle)
        except BaseException:
            await self._release_failed_run(handle)
            raise

    async def _poll_run(self, handle: SubmissionHandle) -> Results:
        """Wait for a run without client callbacks and fetch its results."""
        await self.wait_for_experiment(handle)
        status = await self.get_experiment_status(handle)
        results = await self.get_experiment(handle)
        if status is not SubmissionStatus.COMPLETED:
            error = (
                {"message": results.execution_errors[0][2]}
                if results.execution_errors
                else None
            )
            raise_on_failed_run(status.value, error)
        return results

    async def _serve_run(
        self, handle: SubmissionHandle, callbacks: dict[str, Callable[..., Any]]
    ) -> Results:
        """Serve the client callbacks of a run until it ends and fetch its results."""
        session = ClientCallbackSession(callbacks, self._results_factory(handle))
        done = await self._serve_callbacks(handle, session)
        if session.user_error is not None:
            raise session.user_error
        raise_on_failed_run(done.status, done.error)
        return await self.get_experiment(handle)

    async def _release_failed_run(self, handle: SubmissionHandle) -> None:
        """Release a failed run; the run's own error is what the caller gets to see."""
        try:
            await self.close_submission(handle)
        except Exception:
            _logger.warning(
                "Could not release experiment %s on the controller service after"
                " its run failed",
                handle.hex,
                exc_info=True,
            )

    def _results_factory(self, handle: SubmissionHandle) -> Callable[[], Results]:
        """Build the per-invocation results facade for callback messages."""
        fetch = partial(
            fetch_acquired_results, self._remote_url, self._headers, handle.hex
        )
        fetch_structure = partial(
            fetch_result_structure, self._remote_url, self._headers, handle.hex
        )
        fetch_callbacks = partial(
            fetch_neartime_callback_results,
            self._remote_url,
            self._headers,
            handle.hex,
        )

        def factory() -> Results:
            return Results(
                acquired_results=LazyAcquiredResults(fetch, fetch_structure),
                neartime_callback_results=cast(
                    "dict", LazyNeartimeCallbackResults(fetch_callbacks)
                ),
            )

        return factory

    async def _serve_callbacks(
        self, handle: SubmissionHandle, session: ClientCallbackSession
    ) -> Done:
        """Serve callback messages over the WebSocket until the run ends.

        Reconnects with backoff after a disconnect. Between attempts, the run
        status is probed over REST, so that a run that ended while the socket
        was down terminates the loop. An unreachable service raises instead.
        A channel that another client keeps occupied is given up on after
        `_CHANNEL_BUSY_TIMEOUT_S`.
        """
        url = ws_url(self._remote_url, f"v1/experiments/{handle.hex}/callbacks")
        backoff = _RECONNECT_BACKOFF_INITIAL_S
        busy_deadline: float | None = None
        while True:
            try:
                async with ws_connect(
                    url, additional_headers=self._headers
                ) as websocket:
                    async for raw in websocket:
                        # Messages are flowing, so a later disconnect is transient.
                        backoff = _RECONNECT_BACKOFF_INITIAL_S
                        busy_deadline = None
                        outcome = session.handle_message(json_module.loads(raw))
                        if isinstance(outcome, Done):
                            return outcome
                        if isinstance(outcome, PendingCall):
                            outcome = await session.execute_async(outcome)
                        # stdlib json on purpose: the channel relies on its
                        # tolerance for the non-standard NaN/Infinity literals
                        # in both directions.
                        await websocket.send(
                            json_module.dumps(outcome, default=json_default)
                        )
            except ConnectionClosedError as exc:
                close = exc.rcvd
                if close is not None and close.code == 1008:
                    raise APIError(
                        f"Callback channel rejected: {close.reason}"
                    ) from exc
                if close is not None and close.code == 1013:
                    if busy_deadline is None:
                        busy_deadline = time.monotonic() + _CHANNEL_BUSY_TIMEOUT_S
                    elif time.monotonic() > busy_deadline:
                        raise APIError(
                            "Another client is already serving the near-time callbacks"
                            f" of experiment {handle.hex}: {close.reason}"
                        ) from exc
            except (OSError, InvalidHandshake, ConnectionClosed):
                pass
            status = await self.get_experiment_status(handle)
            if status in _TERMINAL_STATUSES:
                return Done(status=status.value, error=None)
            if backoff >= _RECONNECT_BACKOFF_MAX_S:
                _logger.warning(
                    "Cannot attach the callback channel of experiment %s; retrying"
                    " every %.0f s. The run cannot progress until it attaches.",
                    handle.hex,
                    _RECONNECT_BACKOFF_MAX_S,
                )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, _RECONNECT_BACKOFF_MAX_S)

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        data = await self._request_json("GET", f"v1/experiments/{handle.hex}/status")
        return SubmissionStatus(data["status"])

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        data = await self._request_json("GET", f"v1/experiments/{handle.hex}/results")
        return cast("Results", from_dict(data["results"]))

    async def cancel_experiment(self, handle: SubmissionHandle) -> None:
        await self._request("DELETE", f"v1/experiments/{handle.hex}")

    async def close_submission(self, handle: SubmissionHandle) -> None:
        await self.cancel_experiment(handle)

    async def _request_json(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = await self._request(method, path, json=json)
        if not isinstance(data, dict):
            raise APIError("Unexpected response from server")
        return data

    async def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any] | None:
        url = f"{self._remote_url}/{path.lstrip('/')}"
        # TODO(2K): Consider reusing the client, which however will create a problem
        # of the client lifecycle management and a proper cleanup. For now, we create
        # a new client for each request, which is simpler to manage.
        async with httpx.AsyncClient(transport=self._transport) as client:
            resp = await client.request(method, url, json=json, headers=self._headers)

            if 400 <= resp.status_code < 500:
                raise APIError(f"{resp.status_code}: {resp.text}")
            elif 500 <= resp.status_code < 600:
                reraise_controller_exception(resp)
                raise APIError(f"{resp.status_code}: {resp.text}")

        if resp.content and resp.headers.get("content-type", "").startswith(
            "application/json"
        ):
            return resp.json()
        return None
