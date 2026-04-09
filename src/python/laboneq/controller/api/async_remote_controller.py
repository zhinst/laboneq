# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
        remote_url: str, ignore_version_mismatch: bool | None = None
    ) -> AsyncRemoteController:
        remote_controller = AsyncRemoteController(
            remote_url=remote_url, ignore_version_mismatch=ignore_version_mismatch
        )
        await remote_controller._connect()
        return remote_controller

    def __init__(
        self,
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
    ):
        self._remote_url = remote_url
        self._ignore_version_mismatch = ignore_version_mismatch
        self._headers = {
            "X-LabOneQ-Client-Version": get_version(),
            "X-LabOneQ-Protocol-Version": "1.0",
            "Content-Type": "application/json",
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
        serialized = await self._request("GET", "v1/device-setup")
        device_setup = cast(DeviceSetup, from_dict(serialized))
        return device_setup

    async def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        serialized = to_dict(scheduled_experiment)
        assert isinstance(serialized, dict)  # to satisfy type checker
        handle = SubmissionHandle(handle_id=uuid.uuid4().int)
        await self._request("PUT", f"v1/experiments/{handle.hex}", json=serialized)
        return handle

    async def wait_for_experiment(self, handle: SubmissionHandle):
        raise NotImplementedError

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        raise NotImplementedError

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        raise NotImplementedError

    async def cancel_experiment(self, handle: SubmissionHandle):
        raise NotImplementedError

    async def close_submission(self, handle: SubmissionHandle):
        raise NotImplementedError

    async def _request(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ):
        url = f"{self._remote_url}/{path.lstrip('/')}"
        # TODO(2K): Consider reusing the client, which however will create a problem
        # of the client lifecycle management and a proper cleanup. For now, we create
        # a new client for each request, which is simpler to manage.
        async with httpx.AsyncClient() as client:
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
