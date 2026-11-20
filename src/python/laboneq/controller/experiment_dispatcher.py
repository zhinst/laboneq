# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from laboneq.controller.api.commons import SubmissionRegistry, to_results
from laboneq.controller.round_robin_worker import RoundRobinWorker

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

    from laboneq.controller.api.commons import SubmissionHandle
    from laboneq.controller.callback_channel import CallbackChannel
    from laboneq.controller.controller_core import (
        ControllerCore,
        ControllerSubmission,
        SubmissionStatus,
    )
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.scheduled_experiment import ScheduledExperiment
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.result.results import Results

_logger = logging.getLogger(__name__)


class ExperimentDispatcher:
    """Owns a `ControllerCore` and runs queued experiments on it one at a time.

    Each queue runs its experiments in submission order, and the queues take
    turns. An experiment with a callback channel waits at the front of its queue
    until the client attaches to the channel, while the other queues keep
    running. If the client stays detached for the absence timeout of the
    channel, the experiment ends with an execution error.
    """

    def __init__(self, core: ControllerCore, device_setup: DeviceSetup):
        self._core = core
        self._device_setup = device_setup
        self._submissions = SubmissionRegistry()
        self._worker: RoundRobinWorker[ControllerSubmission] = RoundRobinWorker(
            run_one=core.execute_submission,
            wait_ready=self._wait_for_callback_channel,
            is_ready=_is_callback_channel_attached,
        )

    async def connect(
        self,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ):
        await self._core.connect(
            do_emulation=do_emulation,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
            timeout_s=timeout_s,
        )

    async def submit_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        queue_id: Hashable = None,
        handle: SubmissionHandle | None = None,
        callback_channel: CallbackChannel | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> SubmissionHandle:
        """Queues an experiment on the queue `queue_id` and returns its handle.

        Raises `HardwareNotReadyError` if the instruments cannot run experiments
        now, and `LabOneQControllerException` if `handle` is in use or the
        experiment fails validation.
        """
        handle = self._submissions.ensure_unique_handle(handle)
        submission = await self._core.prepare_submission(
            scheduled_experiment, callback_channel, coprocessor_payload_overrides
        )
        # A concurrent submission may have taken `handle` while the core prepared
        # this one.
        self._submissions.ensure_unique_handle(handle)
        self._submissions.add(handle, submission)
        await self._worker.submit(queue_id, submission)
        return handle

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._core.neartime_callbacks

    async def get_not_ready_reason(self) -> str | None:
        """Returns why the core cannot run experiments now, or None if it can."""
        return await self._core.get_not_ready_reason()

    async def wait_until_ready(self, timeout_s: float):
        """Returns once the core can run experiments.

        Raises `HardwareNotReadyError` after `timeout_s` seconds.
        """
        await self._core.wait_until_ready(timeout_s)

    async def merge_setup_description(
        self, instrument_topology: InstrumentTopology
    ) -> InstrumentTopology:
        """Returns `instrument_topology` with the setup description of the instruments.

        Fetches the setup description if needed. Raises if the core cannot run
        experiments now.
        """
        return await self._core.merge_setup_description(instrument_topology)

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        """Disables the outputs of the given logical signals or instruments.

        Without logical signals, disables every output of the given instruments,
        or of all instruments if none are given. With `unused_only`, keeps the
        outputs that a logical signal uses.
        """
        await self._core.disable_outputs(device_uids, logical_signals, unused_only)

    def get_callback_channel(self, handle: SubmissionHandle) -> CallbackChannel | None:
        return self._submissions.get(handle).callback_channel

    async def wait_for_experiment(self, handle: SubmissionHandle):
        await self._core.wait_submission(self._submissions.get(handle))

    def get_experiment_status(self, handle: SubmissionHandle) -> SubmissionStatus:
        return self._core.submission_status(self._submissions.get(handle))

    def get_experiment_results(self, handle: SubmissionHandle) -> Results:
        """Returns the current, possibly partial, results of the experiment."""
        return to_results(
            self._device_setup,
            self._core.submission_results(self._submissions.get(handle)),
        )

    async def cancel_experiment(self, handle: SubmissionHandle):
        """Drops the experiment if it has not started, otherwise waits for it to end.

        A dropped experiment ends with an execution error.
        """
        # TODO(2K): Stop a running experiment. Only a queued one is dropped.
        self._drop_if_queued(self._submissions.get(handle))
        await self.wait_for_experiment(handle)

    async def close_submission(self, handle: SubmissionHandle):
        await self.cancel_experiment(handle)
        self._submissions.remove(handle)

    async def aclose(self, timeout_s: float | None = None):
        """Stops the dispatcher and disconnects its core.

        Drops the queued experiments. Waits up to `timeout_s` seconds, or without
        limit if None, for a running experiment to end, and stops it with an
        execution error if it has not.
        """
        submissions = [self._submissions.get(handle) for handle in self._submissions]
        try:
            # Dropping every queued experiment before waiting for any keeps the
            # worker from starting one once a running one ends.
            for submission in submissions:
                self._drop_if_queued(submission)
            futures = [submission.completion_future for submission in submissions]
            if futures:
                await asyncio.wait(futures, timeout=timeout_s)
            running = [s for s in submissions if not s.completion_future.done()]
            if running:
                _logger.warning(
                    "Stopping the running experiment after waiting %g s for it.",
                    timeout_s,
                )
                for submission in running:
                    self._core.abort_submission(
                        submission, "The experiment was stopped by a shutdown."
                    )
                await self._worker.cancel()
                await self._core.abort_result_reads()
            await self._worker.stop()
        finally:
            await self._core.disconnect()

    def _drop_if_queued(self, submission: ControllerSubmission):
        if self._worker.discard(submission):
            self._core.abort_submission(
                submission, "The experiment was cancelled before it started."
            )

    async def _wait_for_callback_channel(
        self, submission: ControllerSubmission
    ) -> bool:
        """Waits for the client callback channel of `submission`, if it has one.

        Returns False if the channel stays detached for its absence timeout, after
        ending the submission with an execution error.
        """
        channel = submission.callback_channel
        if channel is None or await channel.wait_attached(restart_window=True):
            return True
        self._core.abort_submission(
            submission,
            f"The client callback channel was not attached within "
            f"{channel.absence_timeout_s:.0f} s of the experiment becoming "
            f"ready to run.",
        )
        return False


def _is_callback_channel_attached(submission: ControllerSubmission) -> bool:
    channel = submission.callback_channel
    return channel is None or channel.is_attached
