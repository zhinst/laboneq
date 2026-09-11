# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any, cast

import httpx

from laboneq._version import get_version
from laboneq.controller.api.async_controller_api import (
    AsyncControllerAPI,
)
from laboneq.controller.api.commons import (
    APIError,
    SubmissionHandle,
    encode_submission,
    reraise_controller_exception,
)
from laboneq.controller.controller import SubmissionStatus
from laboneq.serializers import from_dict

if TYPE_CHECKING:
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


class AsyncRemoteController(AsyncControllerAPI):
    #: Total time to wait for the controller service to become ready on connect.
    _READYZ_TIMEOUT_S: float = 60.0
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
            "X-LabOneQ-Client-Version": get_version(),
            "X-LabOneQ-Protocol-Version": "1.0",
        }
        if ignore_version_mismatch is not None:
            self._headers["X-LabOneQ-Ignore-Version-Mismatch"] = str(
                ignore_version_mismatch
            ).lower()

    async def _connect(self):
        await self._wait_until_ready()

    async def _wait_until_ready(self) -> None:
        """Poll the controller service's `/readyz` endpoint until it is ready.

        The service may still be starting up (connecting to hardware, or
        booting in emulation mode) when the client first tries to reach it;
        retry until it reports ready or `_READYZ_TIMEOUT_S` elapses.
        """
        url = f"{self._remote_url}/readyz"
        deadline = time.monotonic() + self._READYZ_TIMEOUT_S
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
                    f"Controller service at {self._remote_url} did not become "
                    f"ready within {self._READYZ_TIMEOUT_S:.0f}s: {last_error}"
                )
            await asyncio.sleep(self._READYZ_POLL_INTERVAL_S)

    async def aclose(self):
        pass

    async def get_default_devicesetup(self) -> DeviceSetup:
        data = await self._request_json("GET", "v1/devicesetup")
        return cast("DeviceSetup", from_dict(data["device_setup"]))

    async def get_instrument_topology(self) -> InstrumentTopology:
        data = await self._request_json("GET", "v1/instrumenttopology")
        if data.get("instrument_topology") is None:
            raise APIError("Server has no instrument topology configured")
        return cast("InstrumentTopology", from_dict(data["instrument_topology"]))

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
        _TERMINAL = {
            SubmissionStatus.COMPLETED,
            SubmissionStatus.FAILED,
        }
        poll_interval = 0.5
        while True:
            exp_status = await self.get_experiment_status(handle)
            if exp_status in _TERMINAL:
                return
            await asyncio.sleep(poll_interval)

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
