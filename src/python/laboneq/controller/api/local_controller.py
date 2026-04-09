# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from laboneq.controller.api.controller_api import (
    ControllerAPI,
    SubmissionHandle,
    SubmissionStatus,
)
from laboneq.controller.controller import Controller as _Controller
from laboneq.controller.devices.device_collection import DEFAULT_TIMEOUT_S
from laboneq.controller.runtime_context_impl import LegacySessionData
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.result.results import Results
from laboneq.implementation.legacy_adapters.device_setup_converter import (
    convert_device_setup_to_setup,
)
from laboneq.implementation.payload_builder.target_setup_generator import (
    TargetSetupGenerator,
)

if TYPE_CHECKING:
    from laboneq.controller.controller import ControllerSubmission
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.data.scheduled_experiment import ScheduledExperiment


class LocalController(ControllerAPI):
    """Controller interface."""

    @staticmethod
    def create(
        device_setup: DeviceSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable] | None = None,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ) -> LocalController:
        """Create an instance of the Controller."""
        if timeout_s is None:
            timeout_s = DEFAULT_TIMEOUT_S
        setup = convert_device_setup_to_setup(device_setup)
        target_setup = TargetSetupGenerator.from_setup(setup)
        if neartime_callbacks is None:
            neartime_callbacks = {}
        controller = _Controller(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
        )
        controller.start()
        controller.connect(
            do_emulation=do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )
        return LocalController(device_setup=device_setup, controller=controller)

    def __init__(
        self,
        device_setup: DeviceSetup,
        controller: _Controller,
    ):
        self._device_setup = device_setup
        # Keep reference to avoid garbage collection
        self._controller = controller
        self._submissions: dict[int, ControllerSubmission] = {}
        # TODO: Remove _legacy_session_data tests once the RuntimeContext endpoints are removed
        self._legacy_session_data = LegacySessionData(None, None, None, None, None)

    def _submission(self, handle: SubmissionHandle) -> ControllerSubmission:
        submission = self._submissions.get(handle.id)
        if submission is None:
            raise LabOneQControllerException("Invalid submission handle.")
        return submission

    def _get_results(self, handle: SubmissionHandle) -> ExperimentResults:
        submission = self._submission(handle)
        return self._controller.submission_results(submission)

    def close(self):
        for handle_id in self._submissions.keys():
            self.cancel_experiment(SubmissionHandle(handle_id))
        self._controller.disconnect()

    def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._controller._neartime_callbacks.update(neartime_callbacks)

    def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        handle = SubmissionHandle()
        # TODO: Remove _legacy_session_data tests once the RuntimeContext endpoints are removed
        self._controller.set_legacy_session_data(self._legacy_session_data)
        self._submissions[handle.id] = self._controller.submit_compiled(
            scheduled_experiment=scheduled_experiment,
        )
        return handle

    def wait_for_experiment(self, handle: SubmissionHandle):
        self._controller.wait_submission(self._submission(handle))

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        submission = self._submission(handle)
        status = (
            SubmissionStatus.COMPLETED
            if submission.completion_future.done()
            else SubmissionStatus.QUEUED
        )
        if status == SubmissionStatus.COMPLETED:
            results = self._get_results(handle)
            if len(results.execution_errors) > 0:
                status = SubmissionStatus.FAILED
        # TODO(2K): Implement RUNNING status
        return status

    def get_experiment(self, handle: SubmissionHandle) -> Results:
        status = self.get_experiment_status(handle)
        if status not in [SubmissionStatus.COMPLETED, SubmissionStatus.FAILED]:
            self.wait_for_experiment(handle)
        experiment_results = self._get_results(handle)
        return Results(
            device_setup=self._device_setup,
            acquired_results=experiment_results.acquired_results,
            neartime_callback_results=experiment_results.neartime_callback_results,
            execution_errors=experiment_results.execution_errors,
            pipeline_jobs_timestamps=experiment_results.pipeline_jobs_timestamps,
        )

    def cancel_experiment(self, handle: SubmissionHandle):
        self._submission(handle)  # Validate handle
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        self.wait_for_experiment(handle)

    def close_submission(self, handle: SubmissionHandle):
        self.cancel_experiment(handle)
        self._submissions.pop(handle.id)

    # TODO: Remove _legacy_session_data tests once the RuntimeContext endpoints are removed
    def set_legacy_session_data(self, legacy_session_data: LegacySessionData):
        self._legacy_session_data = legacy_session_data
        self._controller.set_legacy_session_data(legacy_session_data)
