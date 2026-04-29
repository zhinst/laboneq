# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import uuid
from typing import Any, cast

import httpx

from laboneq._version import get_version
from laboneq.controller.api.async_controller_api import (
    AsyncControllerAPI,
    SubmissionHandle,
    SubmissionStatus,
)
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.result.results import Results
from laboneq.serializers import from_dict, to_dict


# --- Exceptions ---
class APIError(Exception):
    pass


class NotFound(APIError):
    pass


class Unauthorized(APIError):
    pass


class BadRequest(APIError):
    pass


class ServerError(APIError):
    pass


class AsyncRemoteController(AsyncControllerAPI):
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
        pass

    async def aclose(self):
        pass

    async def get_default_devicesetup(self) -> DeviceSetup:
        data = await self._request_json("GET", "v1/devicesetup")
        return cast(DeviceSetup, from_dict(data["device_setup"]))

    async def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        serialized = to_dict(scheduled_experiment)
        assert isinstance(serialized, dict)
        handle = SubmissionHandle(handle_id=uuid.uuid4().int)
        await self._request(
            "PUT",
            f"v1/experiments/{handle.hex}",
            json=serialized,
        )
        return handle

    async def wait_for_experiment(self, handle: SubmissionHandle) -> None:
        _TERMINAL = {
            SubmissionStatus.COMPLETED,
            SubmissionStatus.FAILED,
            SubmissionStatus.CANCELED,
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
        data = await self._request_json("GET", f"v1/experiments/{handle.hex}")
        if data.get("error"):
            raise APIError(data["error"])
        if data.get("results") is not None:
            return cast(Results, from_dict(data["results"]))
        raise APIError(f"Experiment has no results (status: {data.get('status')})")

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

            # TODO(2K): Proper transfer of server-side controller exceptions, e.g. by defining
            # a common error format and deserializing it here.
            # For now, we just raise generic exceptions based on the status code.
            if 400 <= resp.status_code < 500:
                text = resp.text
                if resp.status_code == 400:
                    raise BadRequest(text)
                if resp.status_code == 401:
                    raise Unauthorized(text)
                if resp.status_code == 404:
                    raise NotFound(text)
                raise APIError(f"{resp.status_code}: {text}")
            elif 500 <= resp.status_code < 600:
                raise ServerError(resp.text)

        if resp.content and resp.headers.get("content-type", "").startswith(
            "application/json"
        ):
            return resp.json()
        return None
