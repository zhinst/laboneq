# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, cast

import httpx

from laboneq import __version__
from laboneq.controller.api.async_controller_api import (
    AsyncControllerAPI,
)
from laboneq.controller.api.callback_client import (
    preflight_client_callbacks,
)
from laboneq.controller.api.commons import (
    APIError,
    ClientCallbackServer,
    SubmissionHandle,
    encode_submission,
    readyz_not_ready_reason,
    reraise_controller_exception,
)
from laboneq.controller.callback_protocol import (
    CALLBACK_PROTOCOL_HEADER,
    CALLBACK_PROTOCOL_VERSION,
)
from laboneq.controller.constants import (
    PROTOCOL_VERSION,
    PROTOCOL_VERSION_HEADER,
    READY_TIMEOUT_S,
)
from laboneq.controller.controller_core import SubmissionStatus
from laboneq.controller.utilities.exception import HardwareNotReadyError
from laboneq.core.exceptions import LabOneQException
from laboneq.serializers import from_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


_TERMINAL_STATUSES = frozenset({SubmissionStatus.COMPLETED, SubmissionStatus.FAILED})


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
        neartime_callbacks: Mapping[str, Callable[..., Any]] | None = None,
        _transport: httpx.AsyncBaseTransport | None = None,
    ) -> AsyncRemoteController:
        """Create an instance of the AsyncRemoteController and connect it.

        For every experiment it submits, the controller serves the near-time
        callbacks in `neartime_callbacks` that the service does not provide,
        until the submission or the controller is closed. It looks a callback up
        when the service calls it, so that callbacks added to the mapping later
        are served as well.
        """
        remote_controller = AsyncRemoteController(
            remote_url=remote_url,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
            _transport=_transport,
        )
        await remote_controller._connect()
        return remote_controller

    def __init__(
        self,
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        neartime_callbacks: Mapping[str, Callable[..., Any]] | None = None,
        _transport: httpx.AsyncBaseTransport | None = None,
    ):
        self._remote_url = remote_url
        self._neartime_callbacks = (
            {} if neartime_callbacks is None else neartime_callbacks
        )
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
        self._callback_server = ClientCallbackServer(
            remote_url, self._headers, self._neartime_callbacks, self._has_ended
        )

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
        await self._callback_server.stop_all()

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

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        raise LabOneQException(
            "disable_outputs() is not supported when connected via a "
            "remote controller service."
        )

    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        client_callbacks = preflight_client_callbacks(
            scheduled_experiment, self._neartime_callbacks
        )
        body = encode_submission(scheduled_experiment, coprocessor_payload_overrides)
        if handle is None:
            handle = SubmissionHandle()
        await self._request(
            "PUT",
            f"v1/experiments/{handle.hex}",
            json=body,
        )
        if client_callbacks:
            await self._callback_server.start(handle)
        return handle

    async def wait_for_experiment(self, handle: SubmissionHandle) -> None:
        await self._callback_server.wait(handle)
        poll_interval = 0.5
        while True:
            exp_status = await self.get_experiment_status(handle)
            if exp_status in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(poll_interval)

    async def _has_ended(self, handle: SubmissionHandle) -> bool:
        return await self.get_experiment_status(handle) in _TERMINAL_STATUSES

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        data = await self._request_json("GET", f"v1/experiments/{handle.hex}/status")
        return SubmissionStatus(data["status"])

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        return await self.get_experiment_results(handle)

    async def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        data = await self._request_json("GET", f"v1/experiments/{handle.hex}/results")
        return cast("Results", from_dict(data["results"]))

    async def cancel_experiment(self, handle: SubmissionHandle) -> None:
        await self._request("DELETE", f"v1/experiments/{handle.hex}")

    async def close_submission(self, handle: SubmissionHandle) -> None:
        await self._callback_server.stop(handle)
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
