# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, cast

import httpx

from laboneq import __version__
from laboneq.controller.api.callback_client import (
    preflight_client_callbacks,
)
from laboneq.controller.api.commons import (
    AUTH_HEADER,
    AUTH_SCHEME,
    APIError,
    ClientCallbackServer,
    SubmissionHandle,
    encode_submission,
    readyz_not_ready_reason,
    reraise_controller_exception,
)
from laboneq.controller.api.controller_api import (
    ControllerAPI,
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
from laboneq.core.utilities.async_helpers import EventLoopHolder
from laboneq.serializers import from_dict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.result.results import Results


_TERMINAL_STATUSES = frozenset({SubmissionStatus.COMPLETED, SubmissionStatus.FAILED})


class RemoteController(ControllerAPI):
    #: Total time to wait for the controller service to come up on connect.
    _LIVEZ_TIMEOUT_S: float = 60.0
    #: Delay between successive liveness polls.
    _LIVEZ_POLL_INTERVAL_S: float = 0.5
    #: Delay between successive readiness polls.
    _READYZ_POLL_INTERVAL_S: float = 0.5

    @staticmethod
    def create(
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        auth_token: str | None = None,
        neartime_callbacks: Mapping[str, Callable[..., Any]] | None = None,
    ) -> RemoteController:
        """Create an instance of the RemoteController and connect it.

        For every experiment it submits, the controller serves the near-time
        callbacks in `neartime_callbacks` that the service does not provide,
        until the submission or the controller is closed. It looks a callback up
        when the service calls it, so that callbacks added to the mapping later
        are served as well. The callbacks run on a thread of the controller.
        """
        remote_controller = RemoteController(
            remote_url=remote_url,
            ignore_version_mismatch=ignore_version_mismatch,
            auth_token=auth_token,
            neartime_callbacks=neartime_callbacks,
        )
        remote_controller._connect()
        return remote_controller

    def __init__(
        self,
        remote_url: str,
        ignore_version_mismatch: bool | None = None,
        auth_token: str | None = None,
        neartime_callbacks: Mapping[str, Callable[..., Any]] | None = None,
    ):
        self._remote_url = remote_url
        self._neartime_callbacks = (
            {} if neartime_callbacks is None else neartime_callbacks
        )
        self._ignore_version_mismatch = ignore_version_mismatch
        self._headers = {
            "X-LabOneQ-Client-Version": __version__,
            PROTOCOL_VERSION_HEADER: PROTOCOL_VERSION,
            CALLBACK_PROTOCOL_HEADER: CALLBACK_PROTOCOL_VERSION,
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
        self._callback_server = ClientCallbackServer(
            remote_url,
            self._headers,
            self._neartime_callbacks,
            partial(asyncio.to_thread, self._has_ended),
        )
        # Runs `_callback_server`. Started by the first submission that has
        # client callbacks to serve.
        self._callback_loop: EventLoopHolder | None = None

    def _connect(self):
        self._wait_until_live()

    def _wait_until_live(self) -> None:
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
                    f"Controller service at {self._remote_url} did not come up "
                    f"within {self._LIVEZ_TIMEOUT_S:.0f}s: {last_error}"
                )
            time.sleep(self._LIVEZ_POLL_INTERVAL_S)

    def close(self):
        if self._callback_loop is None:
            return
        try:
            self._callback_loop.run(self._callback_server.stop_all)
        finally:
            self._callback_loop.stop()
            self._callback_loop = None

    def get_default_devicesetup(self) -> DeviceSetup:
        data = self._request_json("GET", "v1/devicesetup")
        if data.get("device_setup") is None:
            raise APIError("Server has no device setup configured")
        return cast("DeviceSetup", from_dict(data["device_setup"]))

    def get_instrument_topology(self) -> InstrumentTopology | None:
        data = self._request_json("GET", "v1/instrumenttopology")
        if data.get("instrument_topology") is None:
            return None
        return cast("InstrumentTopology", from_dict(data["instrument_topology"]))

    def get_not_ready_reason(self) -> str | None:
        try:
            with httpx.Client() as client:
                resp = client.get(f"{self._remote_url}/readyz", headers=self._headers)
        except httpx.HTTPError as exc:
            return f"Cannot reach the controller service at {self._remote_url}: {exc}"
        return readyz_not_ready_reason(resp)

    def wait_until_ready(self, timeout_s: float = READY_TIMEOUT_S):
        deadline = time.monotonic() + timeout_s
        while (reason := self.get_not_ready_reason()) is not None:
            if time.monotonic() >= deadline:
                raise HardwareNotReadyError(
                    f"Not ready after {timeout_s:g} s. {reason}"
                )
            time.sleep(self._READYZ_POLL_INTERVAL_S)

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        raise LabOneQException(
            "disable_outputs() is not supported when connected via a "
            "remote controller service."
        )

    def submit_experiment(
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
        self._request("PUT", f"v1/experiments/{handle.hex}", json=body)
        if client_callbacks:
            if self._callback_loop is None:
                self._callback_loop = EventLoopHolder()
                self._callback_loop.start()
            self._callback_loop.run(self._callback_server.start, handle)
        return handle

    def wait_for_experiment(self, handle: SubmissionHandle) -> None:
        if self._callback_loop is not None:
            self._callback_loop.run(self._callback_server.wait, handle)
        poll_interval = 0.5
        while True:
            exp_status = self.get_experiment_status(handle)
            if exp_status in _TERMINAL_STATUSES:
                return
            time.sleep(poll_interval)

    def _has_ended(self, handle: SubmissionHandle) -> bool:
        return self.get_experiment_status(handle) in _TERMINAL_STATUSES

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        data = self._request_json("GET", f"v1/experiments/{handle.hex}/status")
        return SubmissionStatus(data["status"])

    def get_experiment(self, handle: SubmissionHandle) -> Results:
        return self.get_experiment_results(handle)

    def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        data = self._request_json("GET", f"v1/experiments/{handle.hex}/results")
        return cast("Results", from_dict(data["results"]))

    def cancel_experiment(self, handle: SubmissionHandle) -> None:
        self._request("DELETE", f"v1/experiments/{handle.hex}")

    def close_submission(self, handle: SubmissionHandle) -> None:
        if self._callback_loop is not None:
            self._callback_loop.run(self._callback_server.stop, handle)
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

    Raises:
        APIError: The controller service has no instrument topology configured.
    """
    controller = RemoteController.create(
        remote_url=remote_url, ignore_version_mismatch=ignore_version_mismatch
    )
    try:
        topology = controller.get_instrument_topology()
    finally:
        controller.close()
    if topology is None:
        raise APIError(
            f"The controller service at {remote_url} has no instrument topology "
            "configured"
        )
    return topology
