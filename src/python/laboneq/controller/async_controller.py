# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from laboneq.controller.controller import Controller
from laboneq.controller.controller_api import (
    ControllerAPI,
    SubmissionHandle,
    SubmissionStatus,
)
from laboneq.controller.devices.device_collection import DEFAULT_TIMEOUT_S
from laboneq.controller.utilities.exception import LabOneQControllerException

if TYPE_CHECKING:
    from laboneq.controller.controller import ControllerSubmission
    from laboneq.data.execution_payload import TargetSetup
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.data.scheduled_experiment import ScheduledExperiment


class AsyncSession: ...


class AsyncController(ControllerAPI):
    """Asynchronous controller interface."""

    @staticmethod
    async def create(
        target_setup: TargetSetup,
        ignore_version_mismatch: bool,
        neartime_callbacks: dict[str, Callable],
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ) -> AsyncController:
        """Create an instance of the AsyncController."""
        if timeout_s is None:
            timeout_s = DEFAULT_TIMEOUT_S
        async_session = AsyncSession()
        controller = Controller(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            neartime_callbacks=neartime_callbacks,
            parent_session=async_session,
        )
        await controller._connect_async(
            do_emulation=do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )
        return AsyncController(async_session=async_session, controller=controller)

    def __init__(
        self,
        async_session: AsyncSession,
        controller: Controller,
    ):
        # Keep reference to avoid garbage collection
        self._async_session = async_session
        self._controller = controller
        self._submissions: dict[int, ControllerSubmission] = {}

    def _submission(self, handle: SubmissionHandle) -> ControllerSubmission:
        submission = self._submissions.get(handle.id)
        if submission is None:
            raise LabOneQControllerException("Invalid submission handle.")
        return submission

    def _get_results(self, handle: SubmissionHandle) -> ExperimentResults:
        submission = self._submission(handle)
        return self._controller.submission_results(submission)

    async def shutdown(self):
        for handle_id in self._submissions.keys():
            await self.cancel_submission(SubmissionHandle(handle_id))
        await self._controller._disconnect_async()

    async def update_neartime_callbacks(self, neartime_callbacks: dict[str, Callable]):
        self._controller._neartime_callbacks.update(neartime_callbacks)

    async def submit_experiment(
        self, scheduled_experiment: ScheduledExperiment
    ) -> SubmissionHandle:
        handle = SubmissionHandle()
        self._submissions[handle.id] = await self._controller._submit_compiled_async(
            scheduled_experiment=scheduled_experiment,
        )
        return handle

    async def wait_for_completion(self, handle: SubmissionHandle):
        await self._controller._wait_submission_async(self._submission(handle))

    async def submission_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        submission = self._submission(handle)
        status = (
            SubmissionStatus.COMPLETED
            if submission.completion_future.done()
            else SubmissionStatus.PENDING
        )
        if status == SubmissionStatus.COMPLETED:
            results = self._get_results(handle)
            if len(results.execution_errors) > 0:
                status = SubmissionStatus.FAILED
        # TODO(2K): Implement RUNNING status
        return status

    async def submission_results(self, handle: SubmissionHandle) -> ExperimentResults:
        status = await self.submission_status(handle)
        if status not in [SubmissionStatus.COMPLETED, SubmissionStatus.FAILED]:
            await self.wait_for_completion(handle)
        return self._get_results(handle)

    async def cancel_submission(self, handle: SubmissionHandle):
        self._submission(handle)  # Validate handle
        # TODO(2K): No mechanism to cancel yet. Implement proper cancellation.
        await self.wait_for_completion(handle)

    async def close_submission(self, handle: SubmissionHandle):
        await self.cancel_submission(handle)
        self._submissions.pop(handle.id)
