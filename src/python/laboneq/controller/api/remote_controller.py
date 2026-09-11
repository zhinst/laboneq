# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import httpx

from laboneq._version import get_version
from laboneq.controller.api.commons import (
    AUTH_HEADER,
    AUTH_SCHEME,
    APIError,
    SubmissionHandle,
    encode_submission,
    reraise_controller_exception,
)
from laboneq.controller.api.controller_api import (
    ControllerAPI,
)
from laboneq.controller.controller import SubmissionStatus
from laboneq.serializers import from_dict

if TYPE_CHECKING:
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


class RemoteController(ControllerAPI):
    #: Total time to wait for the controller service to become ready on connect.
    _READYZ_TIMEOUT_S: float = 60.0
    #: Delay between successive readiness polls.
    _READYZ_POLL_INTERVAL_S: float = 0.5

    @staticmethod
    def create(
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        auth_token: str | None = None,
    ) -> RemoteController:
        remote_controller = RemoteController(
            remote_url=remote_url,
            ignore_version_mismatch=ignore_version_mismatch,
            auth_token=auth_token,
        )
        remote_controller._connect()
        return remote_controller

    def __init__(
        self,
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        auth_token: str | None = None,
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
        # Every request sends these headers, the readiness poll included, so
        # the token needs no handling at the individual call sites.
        if auth_token is not None:
            self._headers[AUTH_HEADER] = f"{AUTH_SCHEME} {auth_token}"

    def _connect(self):
        self._wait_until_ready()

    def _wait_until_ready(self) -> None:
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
                with httpx.Client() as client:
                    resp = client.get(url, headers=self._headers)
                if resp.status_code == 200:
                    return
                if resp.status_code == httpx.codes.UNAUTHORIZED:
                    # The service is up and refusing us, not still starting
                    # up. Waiting out the deadline would not change that.
                    if AUTH_HEADER in self._headers:
                        raise APIError(
                            f"Controller service at {self._remote_url} rejected "
                            f"the authentication token: {resp.text}"
                        )
                    raise APIError(
                        f"Controller service at {self._remote_url} requires "
                        f"authentication that this client was not given: "
                        f"{resp.text}"
                    )
                last_error = f"{resp.status_code}: {resp.text}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise APIError(
                    f"Controller service at {self._remote_url} did not become "
                    f"ready within {self._READYZ_TIMEOUT_S:.0f}s: {last_error}"
                )
            time.sleep(self._READYZ_POLL_INTERVAL_S)

    def close(self):
        pass

    def get_default_devicesetup(self) -> DeviceSetup:
        data = self._request_json("GET", "v1/devicesetup")
        if data.get("device_setup") is None:
            raise APIError("Server has no device setup configured")
        return cast("DeviceSetup", from_dict(data["device_setup"]))

    def get_instrument_topology(self) -> InstrumentTopology:
        data = self._request_json("GET", "v1/instrumenttopology")
        if data.get("instrument_topology") is None:
            raise APIError("Server has no instrument topology configured")
        return cast("InstrumentTopology", from_dict(data["instrument_topology"]))

    def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        handle: SubmissionHandle | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        body = encode_submission(scheduled_experiment, coprocessor_payload_overrides)
        if handle is None:
            handle = SubmissionHandle()
        self._request("PUT", f"v1/experiments/{handle.hex}", json=body)
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
        data = self._request_json("GET", f"v1/experiments/{handle.hex}/results")
        return cast("Results", from_dict(data["results"]))

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


def get_instrument_topology(
    remote_url: str, ignore_version_mismatch: bool | None = None
) -> InstrumentTopology:
    """Retrieve the instrument topology from the controller service at the given address.

    Connects to the controller service, fetches the instrument topology, and closes
    the connection.
    """
    controller = RemoteController.create(
        remote_url=remote_url, ignore_version_mismatch=ignore_version_mismatch
    )
    try:
        return controller.get_instrument_topology()
    finally:
        controller.close()
