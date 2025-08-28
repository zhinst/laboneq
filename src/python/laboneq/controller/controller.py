# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import time
from typing import TYPE_CHECKING, Callable, Generic, TypeVar
from weakref import ref

from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.for_each import for_each_sync
from laboneq.controller.utilities.sweep_params_tracker import SweepParamsTracker
import numpy as np
import zhinst.utils  # type: ignore
from numpy import typing as npt

from laboneq import __version__
from laboneq.controller.devices.async_support import _gather
from laboneq.controller.devices.device_collection import DeviceCollection
from laboneq.controller.devices.device_utils import NodeCollector, zhinst_core_version
from laboneq.controller.devices.device_zi import DeviceBase, DeviceZI
from laboneq.controller.near_time_runner import NearTimeRunner
from laboneq.controller.recipe_processor import (
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
    get_execution_time,
    pre_process_compiled,
)
from laboneq.controller.results import init_empty_result_by_shape
from laboneq.controller.versioning import (
    MINIMUM_SUPPORTED_LABONE_VERSION,
    RECOMMENDED_LABONE_VERSION,
    LabOneVersion,
    SetupCaps,
)
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.utilities.async_helpers import AsyncWorker, EventLoopMixIn
from laboneq.core.utilities.replace_phase_increment import calc_ct_replacement
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements
from laboneq.data.execution_payload import TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import (
    CodegenWaveform,
    ScheduledExperiment,
)

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse


_logger = logging.getLogger(__name__)


# Only recheck for the proper connected state if there was no check since more than
# the below amount of seconds. Important for performance with many small experiments
# executed in a batch.
CONNECT_CHECK_HOLDOFF = 10  # sec


_SessionClass = TypeVar("_SessionClass")


@dataclass
class ControllerSubmission:
    scheduled_experiment: ScheduledExperiment
    completion_future: asyncio.Future[ExperimentResults]
    temp_run_future: asyncio.Future[None]  # TODO(2K): Temporary
    results: ExperimentResults = field(default_factory=lambda: ExperimentResults())


@dataclass
class ExecutionContext:
    submission: ControllerSubmission
    recipe_data: RecipeData


@dataclass
class NtStepResultContext:
    nt_step: NtStepKey | None
    nt_step_result_completed: asyncio.Future[None]
    execution_context: ExecutionContext


class ExperimentRunner(AsyncWorker[ControllerSubmission]):
    def __init__(self, controller: Controller):
        super().__init__()
        self._controller = controller

    async def run_one(self, item: ControllerSubmission):
        await self._controller._run_one_experiment(item)


class ResultCollector(AsyncWorker[NtStepResultContext]):
    def __init__(self, controller: Controller):
        super().__init__()
        self._controller = controller

    async def run_one(self, item: NtStepResultContext):
        await self._controller._collect_nt_step_results(item)


class Controller(EventLoopMixIn, Generic[_SessionClass]):
    def __init__(
        self,
        target_setup: TargetSetup,
        ignore_version_mismatch: bool,
        neartime_callbacks: dict[str, Callable],
        parent_session: _SessionClass,
    ):
        self._ignore_version_mismatch = ignore_version_mismatch
        self._parent_session_ref = ref(parent_session)
        self._do_emulation = True

        _zhinst_core_version = LabOneVersion.from_version_string(zhinst_core_version())
        self._check_zhinst_core_version_support(_zhinst_core_version)
        self._setup_caps = SetupCaps(client_version=_zhinst_core_version)

        self._devices = DeviceCollection(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            setup_caps=self._setup_caps,
        )

        self._last_connect_check_ts: float | None = None

        # Waves which are uploaded to the devices via pulse replacements
        self._current_waves: dict[str, CodegenWaveform] = {}
        self._neartime_callbacks: dict[str, Callable] = (
            {} if neartime_callbacks is None else neartime_callbacks
        )

        self._experiment_runner = ExperimentRunner(self)
        self._result_collector = ResultCollector(self)

        _logger.debug("Controller debug logging is on")
        _logger.info("VERSION: laboneq %s", __version__)

    def _check_zhinst_core_version_support(self, version: LabOneVersion):
        if version < MINIMUM_SUPPORTED_LABONE_VERSION:
            err_msg = f"'zhinst.core' version '{version}' is not supported. We recommend {RECOMMENDED_LABONE_VERSION}."
            raise LabOneQControllerException(err_msg)

    @property
    def setup_caps(self) -> SetupCaps:
        return self._setup_caps

    @property
    def devices(self) -> dict[str, DeviceZI]:
        return self._devices.devices

    async def _prepare_nt_step(
        self,
        recipe_data: RecipeData,
        sweep_params_tracker: SweepParamsTracker,
        user_set_nodes: NodeCollector,
        nt_step: NtStepKey,
    ):
        # Trigger
        await self._devices.for_each(DeviceBase.configure_trigger, recipe_data)

        # NT Sweep parameters
        for param, value in sweep_params_tracker.updated_params():
            recipe_data.attribute_value_tracker.update(param, value)
        await self._devices.for_each(
            DeviceBase.set_nt_step_nodes, recipe_data, user_set_nodes
        )
        recipe_data.attribute_value_tracker.reset_updated()

        # Feedback
        await self._devices.for_each(DeviceBase.configure_feedback, recipe_data)

        # AWG / pipeliner upload
        await self._devices.for_each(DeviceBase.set_before_awg_upload, recipe_data)
        await self._devices.for_each(
            DeviceZI.prepare_artifacts,
            recipe_data=recipe_data,
            nt_step=nt_step,
        )
        await self._devices.for_each(DeviceBase.set_after_awg_upload, recipe_data)

    async def _after_nt_step(self):
        await self._devices.for_each(DeviceBase.update_warning_nodes)
        device_errors = await self._devices.fetch_device_errors()
        if device_errors is not None:
            raise LabOneQControllerException(device_errors)

    async def _wait_execution_to_stop(
        self,
        recipe_data: RecipeData,
        acquisition_type: AcquisitionType,
        rt_execution_info: RtExecutionInfo,
    ):
        min_wait_time, guarded_wait_time = get_execution_time(rt_execution_info)
        if min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            _logger.info("Estimated RT execution time: %.2f s.", min_wait_time)

        target_devs = [
            device for _, device in self._devices.all if isinstance(device, DeviceBase)
        ]
        response_waiters = await _gather(
            *(
                device.make_waiter_for_execution_done(
                    acquisition_type=acquisition_type,
                    with_pipeliner=rt_execution_info.with_pipeliner,
                    timeout_s=guarded_wait_time,
                )
                for device in target_devs
            )
        )
        await _gather(
            *(
                device.emit_start_trigger(
                    with_pipeliner=rt_execution_info.with_pipeliner
                )
                for device in target_devs
            )
        )
        await _gather(
            *(
                device.wait_for_execution_done(
                    response_waiter=response_waiter,
                    timeout_s=guarded_wait_time,
                    min_wait_time=min_wait_time,
                )
                for device, response_waiter in zip(target_devs, response_waiters)
            )
        )

    # TODO(2K): use timeout passed to connect
    async def _execute_one_step(
        self,
        *,
        execution_context: ExecutionContext,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        timeout_s=5.0,
    ) -> asyncio.Future[None]:
        rt_execution_info = recipe_data.rt_execution_info

        nt_step_result_completed = asyncio.get_running_loop().create_future()
        nt_step_result_context = NtStepResultContext(
            nt_step=nt_step,
            nt_step_result_completed=nt_step_result_completed,
            execution_context=execution_context,
        )

        try:
            await self._devices.for_each(
                DeviceBase.wait_for_channels_ready, timeout_s=timeout_s
            )

            await self._devices.for_each(
                DeviceBase.setup_one_step_execution,
                recipe_data=recipe_data,
                nt_step=nt_step,
                with_pipeliner=rt_execution_info.with_pipeliner,
            )

            # This call must happen after the setup_one_step_execution,
            # as result futures are created there.
            await self._result_collector.submit(nt_step_result_context)

            await self._devices.for_each(
                DeviceBase.wait_for_execution_ready,
                with_pipeliner=rt_execution_info.with_pipeliner,
                timeout_s=timeout_s,
            )
            await self._wait_execution_to_stop(
                recipe_data,
                rt_execution_info.acquisition_type,
                rt_execution_info=rt_execution_info,
            )
            await self._devices.for_each(
                DeviceBase.teardown_one_step_execution,
                with_pipeliner=rt_execution_info.with_pipeliner,
            )
        except (asyncio.TimeoutError, TimeoutError) as e:
            raise LabOneQControllerException(
                "Timeout during execution of a near-time step"
            ) from e
        return nt_step_result_completed

    def connect(
        self,
        do_emulation: bool = True,
        reset_devices: bool = False,
        disable_runtime_checks: bool = True,
        timeout_s: float | None = None,
    ):
        # Remember settings for later implicit connect check
        self._do_emulation = do_emulation
        self._devices.set_timeout(timeout_s)
        self._event_loop.run(
            self._connect_async,
            reset_devices=reset_devices,
            disable_runtime_checks=disable_runtime_checks,
        )

    async def _connect_async(
        self, reset_devices: bool = False, disable_runtime_checks: bool = True
    ):
        now = time.monotonic()
        if (
            self._last_connect_check_ts is None
            or now - self._last_connect_check_ts > CONNECT_CHECK_HOLDOFF
        ):
            await self._devices.connect(
                do_emulation=self._do_emulation,
                reset_devices=reset_devices,
                disable_runtime_checks=disable_runtime_checks,
            )
        self._last_connect_check_ts = now

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        self._event_loop.run(
            self._disable_outputs_async, device_uids, logical_signals, unused_only
        )

    async def _disable_outputs_async(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        await self._devices.disable_outputs(device_uids, logical_signals, unused_only)

    def disconnect(self):
        self._event_loop.run(self._disconnect_async)

    async def _disconnect_async(self):
        _logger.info("Disconnecting from all devices and servers...")
        await self._devices.disconnect()
        self._last_connect_check_ts = None
        _logger.info("Successfully disconnected from all devices and servers.")

    def submit_compiled(
        self, scheduled_experiment: ScheduledExperiment
    ) -> ControllerSubmission:
        return self._event_loop.run(
            self._submit_compiled_async,
            scheduled_experiment=scheduled_experiment,
        )

    def wait_submission(self, submission: ControllerSubmission):
        self._event_loop.run(self._wait_submission_async, submission=submission)

    def stop_workers(self):
        self._event_loop.run(self.stop_workers_async)

    def submission_results(self, submission: ControllerSubmission) -> ExperimentResults:
        return submission.results

    async def _submit_compiled_async(
        self,
        scheduled_experiment: ScheduledExperiment,
    ) -> ControllerSubmission:
        submission = ControllerSubmission(
            scheduled_experiment=scheduled_experiment,
            completion_future=asyncio.get_running_loop().create_future(),
            temp_run_future=asyncio.get_running_loop().create_future(),
        )
        await self._experiment_runner.submit(submission)
        return submission

    async def _wait_submission_async(self, submission: ControllerSubmission):
        await submission.completion_future
        await submission.temp_run_future  # TODO(2K): Temporary

    async def stop_workers_async(self):
        await self._experiment_runner.stop()
        await self._result_collector.stop()

    async def _run_one_experiment(self, submission: ControllerSubmission):
        try:
            await self._execute_compiled_async(submission=submission)
        except BaseException as e:
            submission.completion_future.set_exception(e)

    async def _execute_compiled_async(self, submission: ControllerSubmission):
        # Ensure all connect configurations are still valid!
        await self._connect_async()

        recipe_data = pre_process_compiled(
            submission.scheduled_experiment, self._devices
        )
        results = init_empty_result_by_shape(recipe_data)
        submission.results = results

        execution_context = ExecutionContext(
            submission=submission,
            recipe_data=recipe_data,
        )

        async with self._devices.capture_logs():
            try:
                await self._devices.for_each(DeviceBase.reset_to_idle)
                for_each_sync(
                    self._devices.devices.values(),
                    DeviceBase.allocate_resources,
                    recipe_data,
                )
                await self._devices.for_each(DeviceBase.on_experiment_begin)
                await self._devices.for_each(DeviceBase.init_warning_nodes)
                await self._devices.for_each(
                    DeviceBase.apply_initialization, recipe_data
                )
                await self._devices.for_each(
                    DeviceBase.initialize_oscillators, recipe_data
                )

                # Ensure no side effects from the previous execution in the same session
                self._current_waves.clear()

                _logger.info("Starting near-time execution...")
                try:
                    await NearTimeRunner(
                        controller=self,
                        parent_session_ref=self._parent_session_ref,
                        execution_context=execution_context,
                        recipe_data=recipe_data,
                    ).run(recipe_data.execution)
                except AbortExecution:
                    # eat the exception
                    pass

                # In case of an exception in the preceding flow, we skip this finalization step,
                # as the `completion_future` will be set with the exception, that should take precedence.
                nt_step_result_context = NtStepResultContext(
                    nt_step=None,  # Finalize the execution
                    nt_step_result_completed=asyncio.get_running_loop().create_future(),
                    execution_context=execution_context,
                )
                await self._result_collector.submit(nt_step_result_context)
                # TODO(2K): Waiting for the results here is a temporary workaround, as the
                # result collector has to finish before we unsubscribe in `on_experiment_end` in
                # the `finally` clause. For the future, result nodes subscription lifecycle should
                # be outside the experiment execution.
                await nt_step_result_context.execution_context.submission.completion_future

                _logger.info("Finished near-time execution.")
            except LabOneQControllerException:
                # Report device errors if any - it maybe useful to diagnose the original exception
                device_errors = await self._devices.fetch_device_errors()
                if device_errors is not None:
                    _logger.warning(device_errors)
                raise
            finally:
                # Ensure that the experiment run time is not included in the idle timeout for the connection check.
                self._last_connect_check_ts = time.monotonic()
                await self._devices.for_each(DeviceBase.on_experiment_end)
                # TODO(2K): This is another workaround to ensure that the on_experiment_end
                # is called before the test run completes and the nodes touched in `on_experiment_end`
                # are still captured.
                execution_context.submission.temp_run_future.set_result(None)

    def replace_pulse(
        self,
        recipe_data: RecipeData,
        pulse_uid: str | Pulse,
        pulse_or_array: npt.ArrayLike | Pulse,
    ):
        if isinstance(pulse_uid, str):
            for waveform in recipe_data.scheduled_experiment.pulse_map[
                pulse_uid
            ].waveforms.values():
                if any([instance.can_compress for instance in waveform.instances]):
                    raise LabOneQControllerException(
                        f"Pulse replacement on pulses that allow compression not "
                        f"allowed. Pulse {pulse_uid}"
                    )

        if getattr(pulse_uid, "can_compress", False):
            raise LabOneQControllerException(
                f"Pulse replacement on pulses that allow compression not allowed. "
                f"Pulse {pulse_uid.uid}"  # type: ignore
            )

        acquisition_type = recipe_data.rt_execution_info.acquisition_type
        wave_replacements = calc_wave_replacements(
            recipe_data.scheduled_experiment,
            pulse_uid,
            pulse_or_array,
            self._current_waves,
        )
        for repl in wave_replacements:
            awg_indices = next(
                a
                for a in recipe_data.scheduled_experiment.wave_indices
                if a["filename"] == repl.awg_id
            )
            awg_wave_map: dict[str, list[int | str]] = awg_indices["value"]
            target_wave_index = awg_wave_map[repl.sig_string][0]
            assert isinstance(target_wave_index, int)
            seqc_name = repl.awg_id
            awg_key = recipe_data.awg_by_seqc_name(seqc_name)
            assert awg_key is not None
            device = self._devices.find_by_uid(awg_key.device_uid)

            if repl.replacement_type == ReplacementType.I_Q:
                assert isinstance(repl.samples, list)
                clipped = np.clip(repl.samples, -1.0, 1.0)
                bin_wave = zhinst.utils.convert_awg_waveform(*clipped)
                device.add_waveform_replacement(
                    awg_index=awg_key.awg_index,
                    wave=WaveformItem(
                        index=target_wave_index,
                        name=repl.sig_string + " (repl)",
                        samples=bin_wave,
                    ),
                    acquisition_type=acquisition_type,
                )
            elif repl.replacement_type == ReplacementType.COMPLEX:
                assert isinstance(repl.samples, np.ndarray)
                np.clip(repl.samples.real, -1.0, 1.0, out=repl.samples.real)
                np.clip(repl.samples.imag, -1.0, 1.0, out=repl.samples.imag)
                device.add_waveform_replacement(
                    awg_index=awg_key.awg_index,
                    wave=WaveformItem(
                        index=target_wave_index,
                        name=repl.sig_string + " (repl)",
                        samples=repl.samples,
                    ),
                    acquisition_type=acquisition_type,
                )

    def replace_phase_increment(
        self, recipe_data: RecipeData, parameter_uid: str, new_value: int | float
    ):
        ct_replacements = calc_ct_replacement(
            recipe_data.scheduled_experiment, parameter_uid, new_value
        )
        for repl in ct_replacements:
            seqc_name = repl["seqc"]
            awg_key = recipe_data.awg_by_seqc_name(seqc_name)
            assert awg_key is not None
            device = self._devices.find_by_uid(awg_key.device_uid)
            device.add_command_table_replacement(awg_key.awg_index, repl["ct"])

    async def _collect_nt_step_results(
        self, nt_step_result_context: NtStepResultContext
    ):
        if nt_step_result_context.nt_step is None:
            nt_step_result_context.execution_context.submission.completion_future.set_result(
                nt_step_result_context.execution_context.submission.results
            )
        else:
            await self._read_one_step_results(
                recipe_data=nt_step_result_context.execution_context.recipe_data,
                results=nt_step_result_context.execution_context.submission.results,
                nt_step=nt_step_result_context.nt_step,
            )
        # Indicate that the results for this step are ready.
        nt_step_result_context.nt_step_result_completed.set_result(None)

    async def _read_one_step_results(
        self, recipe_data: RecipeData, results: ExperimentResults, nt_step: NtStepKey
    ):
        await self._devices.for_each(
            DeviceZI.read_results,
            recipe_data=recipe_data,
            nt_step=nt_step,
            results=results,
        )

    def _report_step_error(
        self, results: ExperimentResults, nt_step: NtStepKey, uid: str, message: str
    ):
        results.execution_errors.append((list(nt_step.indices), uid, message))
