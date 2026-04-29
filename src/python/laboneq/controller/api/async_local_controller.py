# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from laboneq.controller.api.async_controller_api import (
    AsyncControllerAPI,
    SubmissionHandle,
    SubmissionStatus,
)
from laboneq.controller.controller import Controller
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


class AsyncLocalController(AsyncControllerAPI):
    """Asynchronous controller interface."""

    @staticmethod
    async def create(
        device_setup: DeviceSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable] | None = None,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ) -> AsyncLocalController:
        """Create an instance of the AsyncLocalController."""
        if timeout_s is None:
            timeout_s = DEFAULT_TIMEOUT_S
        setup = convert_device_setup_to_setup(device_setup)
        target_setup = TargetSetupGenerator.from_setup(setup)
        if neartime_callbacks is None:
            neartime_callbacks = {}
        controller = Controller(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
        )
        await controller._connect_async(
            do_emulation=do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )
        return AsyncLocalController(device_setup=device_setup, controller=controller)

    def __init__(
        self,
        device_setup: DeviceSetup,
        controller: Controller,
    ):
        self._device_setup = device_setup
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

    async def aclose(self):
        for handle_id in self._submissions.keys():
            await self.cancel_experiment(SubmissionHandle(handle_id))
        await self._controller._disconnect_async()

    async def get_default_devicesetup(self) -> DeviceSetup:
        return self._device_setup

    async def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        """Update the neartime callbacks used by the controller.

        This is only present to support the legacy session, don't use it in new code.
        """
        self._controller._neartime_callbacks.update(neartime_callbacks)

    async def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        handle = SubmissionHandle()
        # TODO: Remove _legacy_session_data tests once the RuntimeContext endpoints are removed
        self._controller.set_legacy_session_data(self._legacy_session_data)
        self._submissions[handle.id] = await self._controller._submit_compiled_async(
            scheduled_experiment=scheduled_experiment,
        )
        return handle

    async def wait_for_experiment(self, handle: SubmissionHandle):
        await self._controller._wait_submission_async(self._submission(handle))

    async def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
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

    async def get_experiment(self, handle: SubmissionHandle) -> Results:
        status = await self.get_experiment_status(handle)
        if status not in [SubmissionStatus.COMPLETED, SubmissionStatus.FAILED]:
            await self.wait_for_experiment(handle)
        experiment_results = self._get_results(handle)
        return Results(
            device_setup=self._device_setup,
            acquired_results=experiment_results.acquired_results,
            neartime_callback_results=experiment_results.neartime_callback_results,
            execution_errors=experiment_results.execution_errors,
            pipeline_jobs_timestamps=experiment_results.pipeline_jobs_timestamps,
        )

    async def cancel_experiment(self, handle: SubmissionHandle):
        self._submission(handle)  # Validate handle
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        await self.wait_for_experiment(handle)

    async def close_submission(self, handle: SubmissionHandle):
        await self.cancel_experiment(handle)
        self._submissions.pop(handle.id)

    # TODO: Remove _legacy_session_data tests once the RuntimeContext endpoints are removed
    def set_legacy_session_data(self, legacy_session_data: LegacySessionData):
        self._legacy_session_data = legacy_session_data
        self._controller.set_legacy_session_data(legacy_session_data)
