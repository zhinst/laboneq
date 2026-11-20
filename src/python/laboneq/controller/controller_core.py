# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import attrs

from laboneq import __version__
from laboneq.controller.constants import CONNECT_CHECK_HOLDOFF
from laboneq.controller.controller_backend import (
    StepSubmission,
    create_controller_backend,
)
from laboneq.controller.devices.device_utils import labone_client_version
from laboneq.controller.near_time_replacement import NearTimeReplacements
from laboneq.controller.near_time_runner import NearTimeRunner
from laboneq.controller.recipe_processor import validate_scheduled_experiment
from laboneq.controller.results import ResultsBuilder, error_message
from laboneq.controller.toolkit_adapter import toolkit_instruments
from laboneq.controller.utilities.exception import (
    HardwareNotReadyError,
    LabOneQControllerException,
    RecoverableExecutionError,
)
from laboneq.controller.versioning import (
    RECOMMENDED_MINIMUM_LABONE_VERSION,
    SetupCaps,
)
from laboneq.core.exceptions import AbortExecution
from laboneq.core.utilities.async_helpers import AsyncWorker

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq.controller.backend_runtime_state import BackendRuntimeState
    from laboneq.controller.callback_channel import CallbackChannel, DoneStatus
    from laboneq.controller.devices.device_utils import NodeCollector
    from laboneq.controller.recipe_processor import (
        RecipeData,
    )
    from laboneq.controller.toolkit_adapter import ToolkitInstrument
    from laboneq.controller.utilities.sweep_params_tracker import SweepParamsTracker
    from laboneq.controller.versioning import LabOneVersion
    from laboneq.data.experiment_results import ExperimentResults
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.data.nt_step_key import NtStepKey
    from laboneq.data.scheduled_experiment import (
        ScheduledExperiment,
    )
    from laboneq.data.setup_descriptions import SetupDescription
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment.pulse import Pulse


_logger = logging.getLogger(__name__)


@dataclass
class ControllerSubmission:
    scheduled_experiment: ScheduledExperiment
    completion_future: asyncio.Future[ResultsBuilder]
    results_builder: ResultsBuilder
    callback_channel: CallbackChannel | None = None
    coprocessor_payload_overrides: dict[str, bytes] = field(default_factory=dict)
    # Set once the terminal message went to the callback channel.
    outcome_reported: bool = False


def _complete_submission(submission: ControllerSubmission) -> None:
    """Resolves the submission with its results and reports the outcome to the client."""
    # The SIGINT handler may have cancelled `completion_future`.
    if not submission.completion_future.done():
        submission.completion_future.set_result(submission.results_builder)
    _notify_callback_channel_done(submission)


def _notify_callback_channel_done(submission: ControllerSubmission) -> None:
    """Sends the terminal channel message for how the run ended.

    Later calls send nothing.
    """
    channel = submission.callback_channel
    if channel is None or submission.outcome_reported:
        return
    submission.outcome_reported = True
    future = submission.completion_future
    error: dict[str, Any] | None = None
    if future.cancelled():
        status: DoneStatus = "cancelled"
    elif not future.done() or future.exception() is not None:
        status = "failed"
        if future.done() and (exc := future.exception()) is not None:
            error = {"code": "CONTROLLER_ERROR", "message": str(exc)}
    elif submission.results_builder.results.execution_errors:
        status = "failed"
        error = {
            "code": "EXECUTION_ERROR",
            "message": submission.results_builder.results.execution_errors[0][2],
        }
    else:
        status = "completed"
    channel.notify_done(status, error)


class SubmissionStatus(Enum):
    """The status of an experiment submission."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExecutionContext:
    submission: ControllerSubmission
    recipe_data: RecipeData
    backend_runtime_state: BackendRuntimeState | None = None
    # Whether any near-time step has reached the instruments.
    anything_submitted: bool = False


@dataclass
class NtStepResultContext:
    execution_context: ExecutionContext
    # None for the item that finalizes the run.
    step_submission: StepSubmission | None = None
    # Resolves once the step's results are read. None for the finalizing item.
    nt_step_result_completed: asyncio.Future[None] | None = None

    @classmethod
    def finalizing(cls, execution_context: ExecutionContext) -> NtStepResultContext:
        """Returns the item that makes the result collector finalize the run."""
        return cls(execution_context=execution_context)


class _NothingSubmitted(Exception):
    """The first near-time step reached no instrument; the run cannot proceed."""

    def __init__(self, step_submission: StepSubmission):
        super().__init__(
            "The first near-time step reached no instrument, as the errors of its "
            "submission report. Execution stopped."
        )
        self.step_submission = step_submission


class ControllerCore:
    """Runs experiments on the instruments of one device setup.

    All its coroutines must run on one event loop, because its futures, tasks
    and queues are bound to the loop they were created on.
    """

    #: Minimum time between connect attempts while the instruments are not connected.
    _RECONNECT_HOLDOFF_S: float = CONNECT_CHECK_HOLDOFF
    #: Delay between successive readiness checks in `wait_until_ready`.
    _READY_POLL_INTERVAL_S: float = 0.1

    def __init__(
        self,
        device_setup: DeviceSetup,
        ignore_version_mismatch: bool,
        neartime_callbacks: dict[str, Callable],
    ):
        self._ignore_version_mismatch = ignore_version_mismatch
        self._do_emulation = True
        self._connect_error: Exception | None = None
        self._reconnect_task: asyncio.Task[None] | None = None
        self._last_connect_attempt_s = 0.0
        self._reset_pending = False
        self._disable_runtime_checks = True
        self._timeout_s: float | None = None

        client_version = labone_client_version()
        if client_version is not None:
            self._check_zhinst_core_version_support(client_version)
        self._setup_caps = SetupCaps(client_version=client_version)

        self._backend = create_controller_backend(
            device_setup=device_setup,
            setup_caps=self._setup_caps,
            ignore_version_mismatch=ignore_version_mismatch,
        )
        self._toolkit_instruments = toolkit_instruments(device_setup)

        self._neartime_replacements: NearTimeReplacements = NearTimeReplacements()
        self._neartime_callbacks: dict[str, Callable[..., Any]] = (
            {} if neartime_callbacks is None else neartime_callbacks
        )

        self._result_collector = AsyncWorker(run_one=self._collect_nt_step_results)

        _logger.debug("Controller debug logging is on")
        _logger.info("VERSION: laboneq %s", __version__)

    @property
    def neartime_callbacks(self) -> dict[str, Callable]:
        return self._neartime_callbacks

    def _check_zhinst_core_version_support(self, version: LabOneVersion):
        if version < RECOMMENDED_MINIMUM_LABONE_VERSION:
            err_msg = (
                f"'zhinst.core' version '{version}' is not supported. "
                f"We recommend {RECOMMENDED_MINIMUM_LABONE_VERSION.show(omit_build=True)}."
            )
            raise LabOneQControllerException(err_msg)

    @property
    def setup_caps(self) -> SetupCaps:
        return self._setup_caps

    @property
    def toolkit_instruments(self) -> dict[str, ToolkitInstrument]:
        return self._toolkit_instruments

    @property
    def setup_description(self) -> SetupDescription | None:
        """Setup description from connected device, if any."""
        return self._backend.setup_description

    async def get_not_ready_reason(self) -> str | None:
        """Return why the instruments cannot run experiments now, or None if they can.

        If the last connect failed, starts another attempt in the background and
        reports the failure until an attempt succeeds.
        """
        self._start_reconnect_if_needed()
        if self._connect_error is not None:
            return f"Not connected to the instruments: {self._connect_error}"
        return self._backend.not_ready_reason

    async def wait_until_ready(self, timeout_s: float):
        deadline = time.monotonic() + timeout_s
        while (reason := await self.get_not_ready_reason()) is not None:
            if time.monotonic() >= deadline:
                raise HardwareNotReadyError(
                    f"Not ready after {timeout_s:g} s. {reason}"
                )
            await asyncio.sleep(self._READY_POLL_INTERVAL_S)

    async def _raise_if_not_ready_async(self):
        not_ready_reason = await self.get_not_ready_reason()
        if not_ready_reason is not None:
            raise HardwareNotReadyError(not_ready_reason)

    async def _prepare_nt_step(
        self,
        recipe_data: RecipeData,
        sweep_params_tracker: SweepParamsTracker,
        user_set_nodes: NodeCollector,
        nt_step: NtStepKey,
    ):
        for param, value in sweep_params_tracker.updated_params():
            recipe_data.attribute_value_tracker.update(param, value)
        await self._backend.prepare_nt_step(
            recipe_data=recipe_data,
            nt_step=nt_step,
            user_set_nodes=user_set_nodes,
            replacements=self._neartime_replacements,
        )
        recipe_data.attribute_value_tracker.reset_updated()
        sweep_params_tracker.clear_for_next_step()
        user_set_nodes.clear()

    # TODO(2K): use timeout passed to connect
    async def _execute_one_step(
        self,
        *,
        execution_context: ExecutionContext,
        sweep_params_tracker: SweepParamsTracker,
        user_set_nodes: NodeCollector,
        nt_step: NtStepKey,
        timeout_s=5.0,
    ) -> asyncio.Future[None]:
        recipe_data = execution_context.recipe_data

        try:
            await self._prepare_nt_step(
                recipe_data=recipe_data,
                sweep_params_tracker=sweep_params_tracker,
                user_set_nodes=user_set_nodes,
                nt_step=nt_step,
            )
        except RecoverableExecutionError as e:
            step_submission = StepSubmission.not_submitted(nt_step, e)
        else:
            step_submission = await self._backend.execute_nt_step(
                recipe_data=recipe_data, nt_step=nt_step, timeout_s=timeout_s
            )
        if step_submission.anything_submitted:
            execution_context.anything_submitted = True
        elif not execution_context.anything_submitted:
            raise _NothingSubmitted(step_submission)
        nt_step_result_completed = asyncio.get_running_loop().create_future()
        await self._result_collector.submit(
            NtStepResultContext(
                execution_context=execution_context,
                step_submission=step_submission,
                nt_step_result_completed=nt_step_result_completed,
            )
        )
        try:
            await self._backend.after_nt_step()
        except RecoverableExecutionError as e:
            execution_context.submission.results_builder.add_step_error(
                nt_step, None, "", error_message(e)
            )
        return nt_step_result_completed

    async def connect(
        self,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ):
        # Remember settings for later implicit connect check
        self._do_emulation = do_emulation
        self._reset_pending = reset_devices
        self._disable_runtime_checks = disable_runtime_checks
        self._timeout_s = timeout_s
        await self._attempt_connect_async()

    def _start_reconnect_if_needed(self):
        """Start a connect attempt in the background if the last connect failed.

        Starts none while an attempt runs, or within `_RECONNECT_HOLDOFF_S` of
        the end of the last one.
        """
        if (
            self._connect_error is None
            or (self._reconnect_task is not None and not self._reconnect_task.done())
            or time.monotonic() - self._last_connect_attempt_s
            < self._RECONNECT_HOLDOFF_S
        ):
            return
        self._reconnect_task = asyncio.get_running_loop().create_task(
            self._reconnect_async()
        )

    async def _reconnect_async(self):
        try:
            await self._attempt_connect_async()
        except Exception as e:
            _logger.warning("Cannot connect to the instruments: %s", e)
            return
        _logger.info("Connected to the instruments")

    async def _attempt_connect_async(self):
        try:
            await self._backend.connect(
                do_emulation=self._do_emulation,
                reset_devices=self._reset_pending,
                disable_runtime_checks=self._disable_runtime_checks,
                timeout_s=self._timeout_s,
            )
        except Exception as e:
            self._connect_error = e
            raise
        finally:
            self._last_connect_attempt_s = time.monotonic()
        self._connect_error = None
        # A requested reset is a one-time action.
        self._reset_pending = False

    async def ensure_setup_description(self):
        await self._raise_if_not_ready_async()
        await self._backend.ensure_setup_description()

    async def merge_setup_description(
        self, instrument_topology: InstrumentTopology
    ) -> InstrumentTopology:
        """Returns `instrument_topology` with the setup description of the instruments.

        Fetches the setup description if needed. Returns `instrument_topology`
        unchanged under emulation, or if the instruments have no setup
        description. Raises `HardwareNotReadyError` if the instruments cannot
        run experiments now.
        """
        await self.ensure_setup_description()
        # A client rebuilds a `DeviceSetup` from the topology, and compiles for it
        # with the setup description. Emulation has no instruments to fetch one from.
        setup_description = self.setup_description
        if self._do_emulation or setup_description is None:
            return instrument_topology
        return attrs.evolve(instrument_topology, setup_description=setup_description)

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        await self._backend.disable_outputs(device_uids, logical_signals, unused_only)

    async def disconnect(self):
        await self.stop_workers()
        _logger.info("Disconnecting from all devices and servers...")
        if self._reconnect_task is not None:
            self._reconnect_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reconnect_task
            self._reconnect_task = None
        await self._backend.disconnect()
        _logger.info("Successfully disconnected from all devices and servers.")

    def submission_results(self, submission: ControllerSubmission) -> ExperimentResults:
        return submission.results_builder.results

    def submission_status(self, submission: ControllerSubmission) -> SubmissionStatus:
        if not submission.completion_future.done():
            # TODO(2K): Implement RUNNING status
            return SubmissionStatus.QUEUED
        if len(self.submission_results(submission).execution_errors) > 0:
            return SubmissionStatus.FAILED
        return SubmissionStatus.COMPLETED

    async def prepare_submission(
        self,
        scheduled_experiment: ScheduledExperiment,
        callback_channel: CallbackChannel | None = None,
        coprocessor_payload_overrides: dict[str, bytes] | None = None,
    ) -> ControllerSubmission:
        """Validates `scheduled_experiment` and returns a submission for it.

        Nothing runs until a caller passes the submission to `execute_submission`.
        """
        await self._raise_if_not_ready_async()
        validate_scheduled_experiment(
            scheduled_experiment,
            self._backend.device_setup_fingerprint,
            self._neartime_callbacks,
            callback_channel=callback_channel,
        )
        submission = ControllerSubmission(
            scheduled_experiment=scheduled_experiment,
            completion_future=asyncio.get_running_loop().create_future(),
            results_builder=ResultsBuilder.from_scheduled_experiment(
                scheduled_experiment
            ),
            callback_channel=callback_channel,
            coprocessor_payload_overrides=dict(coprocessor_payload_overrides or {}),
        )
        return submission

    async def wait_submission(self, submission: ControllerSubmission):
        await submission.completion_future

    def abort_submission(self, submission: ControllerSubmission, reason: str) -> None:
        """Ends `submission` with `reason` recorded as its execution error."""
        submission.results_builder.add_execution_error(([], "", reason))
        _complete_submission(submission)

    async def stop_workers(self):
        await self._result_collector.stop()

    async def abort_result_reads(self):
        """Stops reading the results of the submitted near-time steps.

        Drops the queued reads. A submission with a read in progress or queued
        stays unfinished.
        """
        await self._result_collector.cancel()

    async def execute_submission(self, submission: ControllerSubmission):
        """Runs `submission` to its end, and records errors in its results."""
        try:
            await self._execute_compiled_async(submission=submission)
        except Exception as e:
            # Raised before the near-time loop ended, so it stopped the run.
            submission.results_builder.add_fatal_error(([], "", error_message(e)))
            _complete_submission(submission)
        except BaseException:
            # A cancelled run queues no finalizing item, so report its outcome here.
            _notify_callback_channel_done(submission)
            raise

    async def _execute_compiled_async(self, submission: ControllerSubmission):
        recipe_data = self._backend.pre_process(
            submission.scheduled_experiment,
            submission.coprocessor_payload_overrides,
        )

        execution_context = ExecutionContext(
            submission=submission,
            recipe_data=recipe_data,
            backend_runtime_state=self._backend.create_backend_runtime_state(
                recipe_data
            ),
        )

        loop_ended = False
        nothing_submitted = False
        try:
            async with self._backend.experiment(recipe_data):
                # Ensure no side effects from the previous execution in the same session
                self._neartime_replacements.clear()

                _logger.info("Starting near-time execution...")
                runner = NearTimeRunner(
                    controller=self,
                    execution_context=execution_context,
                    do_emulation=self._do_emulation,
                )
                try:
                    await runner.run(recipe_data.execution)
                except AbortExecution:
                    # eat the exception
                    pass
                except _NothingSubmitted as e:
                    submission.results_builder.add_fatal_error(
                        (list(e.step_submission.nt_step.indices), "", str(e))
                    )
                    _record_chunk_errors(submission.results_builder, e.step_submission)
                    nothing_submitted = True
                except Exception as e:
                    submission.results_builder.add_fatal_error(
                        (list(runner.nt_loop_indices), "", error_message(e))
                    )
                loop_ended = True

                _logger.info("Finished near-time execution.")
        except Exception as e:
            if not loop_ended:
                raise
            # The near-time loop has ended, so this error did not stop the run and
            # is not its fatal error.
            submission.results_builder.add_execution_error(([], "", error_message(e)))

        if nothing_submitted:
            _complete_submission(submission)
            return
        # Finalizing completes the submission, so it is queued only after the
        # experiment has exited and any error of the exit is recorded.
        await self._result_collector.submit(
            NtStepResultContext.finalizing(execution_context)
        )

    def register_pulse_replacement(
        self,
        pulse_uid: str,
        replacement: npt.ArrayLike | Pulse,
    ):
        self._neartime_replacements.add_pulse_replacement(pulse_uid, replacement)

    def register_phase_increment_replacement(
        self, parameter_uid: str, new_value: int | float
    ):
        self._neartime_replacements.add_phase_increment_replacement(
            parameter_uid, new_value
        )

    async def _collect_nt_step_results(
        self, nt_step_result_context: NtStepResultContext
    ):
        execution_context = nt_step_result_context.execution_context
        try:
            step_submission = nt_step_result_context.step_submission
            if step_submission is None:
                await self._finalize_results(execution_context)
            else:
                await self._collect_one_step(execution_context, step_submission)
        finally:
            if nt_step_result_context.nt_step_result_completed is not None:
                nt_step_result_context.nt_step_result_completed.set_result(None)

    async def _collect_one_step(
        self, execution_context: ExecutionContext, step_submission: StepSubmission
    ):
        # An error of one step leaves the results of the other steps intact.
        results_builder = execution_context.submission.results_builder
        nt_step = step_submission.nt_step
        _record_chunk_errors(results_builder, step_submission)
        if step_submission.anything_submitted:
            try:
                await self._read_one_step_results(
                    recipe_data=execution_context.recipe_data,
                    results_builder=results_builder,
                    step_submission=step_submission,
                )
            except Exception as e:
                results_builder.add_step_error(nt_step, None, "", error_message(e))
        try:
            await self._backend.process_step_results(
                recipe_data=execution_context.recipe_data,
                results_builder=results_builder,
                backend_runtime_state=execution_context.backend_runtime_state,
                nt_step=nt_step,
            )
        except Exception as e:
            results_builder.add_step_error(nt_step, None, "", error_message(e))

    async def _finalize_results(self, execution_context: ExecutionContext):
        submission = execution_context.submission
        try:
            await self._backend.process_step_results(
                recipe_data=execution_context.recipe_data,
                results_builder=submission.results_builder,
                backend_runtime_state=execution_context.backend_runtime_state,
                nt_step=None,
            )
        except Exception as e:
            submission.results_builder.add_execution_error(([], "", error_message(e)))
        finally:
            # Runs on cancellation too, so a cancelled collector still completes
            # the submission.
            _complete_submission(submission)

    async def _read_one_step_results(
        self,
        recipe_data: RecipeData,
        results_builder: ResultsBuilder,
        step_submission: StepSubmission,
    ):
        await self._backend.read_results(
            recipe_data=recipe_data,
            step_submission=step_submission,
            results_builder=results_builder,
        )


def _record_chunk_errors(
    results_builder: ResultsBuilder, step_submission: StepSubmission
):
    for chunk in step_submission.chunks:
        if chunk.error is not None:
            results_builder.add_step_error(
                step_submission.nt_step,
                chunk.chunk_index,
                "real-time-loop",
                error_message(chunk.error),
            )
