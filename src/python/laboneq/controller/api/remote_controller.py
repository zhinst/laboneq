# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import httpx

from laboneq._version import get_version
from laboneq.controller.api.commons import (
    APIError,
    SubmissionHandle,
    reraise_controller_exception,
)
from laboneq.controller.api.controller_api import (
    ControllerAPI,
)
from laboneq.controller.controller import SubmissionStatus
from laboneq.serializers import from_dict, to_dict

if TYPE_CHECKING:
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


class RemoteController(ControllerAPI):
    @staticmethod
    def create(
        remote_url: str, ignore_version_mismatch: bool | None = None
    ) -> RemoteController:
        remote_controller = RemoteController(
            remote_url=remote_url, ignore_version_mismatch=ignore_version_mismatch
        )
        remote_controller._connect()
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

    def _connect(self):
        pass

    def close(self):
        pass

    def get_default_devicesetup(self) -> DeviceSetup:
        data = self._request_json("GET", "v1/devicesetup")
        if data.get("device_setup") is None:
            raise APIError("Server has no device setup configured")
        return cast("DeviceSetup", from_dict(data["device_setup"]))

    def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
    ) -> SubmissionHandle:
        serialized = to_dict(scheduled_experiment)
        assert isinstance(serialized, dict)  # to satisfy type checker
        if handle is None:
            handle = SubmissionHandle()
        self._request("PUT", f"v1/experiments/{handle.hex}", json=serialized)
        return handle

    def wait_for_experiment(self, handle: SubmissionHandle) -> None:
        _TERMINAL = {
            SubmissionStatus.COMPLETED,
            SubmissionStatus.FAILED,
        }
        poll_interval = 0.5
        while True:
            exp_status = self.get_experiment_status(handle)
            if exp_status in _TERMINAL:
                return
            time.sleep(poll_interval)

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        data = self._request_json("GET", f"v1/experiments/{handle.hex}/status")
        return SubmissionStatus(data["status"])

    def get_experiment(self, handle: SubmissionHandle) -> Results:
        data = self._request_json("GET", f"v1/experiments/{handle.hex}")
        if data.get("results") is not None:
            return cast("Results", from_dict(data["results"]))
        raise APIError(f"Experiment has no results (status: {data.get('status')})")

    def cancel_experiment(self, handle: SubmissionHandle) -> None:
        self._request("DELETE", f"v1/experiments/{handle.hex}")

    def close_submission(self, handle: SubmissionHandle) -> None:
        self.cancel_experiment(handle)

    def _request_json(
        self, method: str, path: str, json: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Like _request, but raises APIError if the response is not a JSON object."""
        data = self._request(method, path, json=json)
        if not isinstance(data, dict):
            raise APIError("Unexpected response from server")
        return data

    def _request(self, method: str, path: str, json: dict[str, Any] | None = None):
        url = f"{self._remote_url}/{path.lstrip('/')}"
        # TODO(2K): Consider reusing the client, which however will create a problem
        # of the client lifecycle management and a proper cleanup. For now, we create
        # a new client for each request, which is simpler to manage.
        with httpx.Client() as client:
            resp = client.request(method, url, json=json, headers=self._headers)

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
