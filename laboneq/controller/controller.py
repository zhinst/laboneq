# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import logging
import time
from collections import defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import zhinst.utils
from numpy import typing as npt

from laboneq import __version__
from laboneq._observability import tracing
from laboneq.controller.communication import (
    DaqNodeSetAction,
    batch_set,
    batch_set_multiple,
)
from laboneq.controller.devices.async_support import gather_and_apply
from laboneq.controller.devices.device_collection import DeviceCollection
from laboneq.controller.devices.device_utils import NodeCollector, zhinst_core_version
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.devices.zi_node_monitor import ResponseWaiter
from laboneq.controller.near_time_runner import NearTimeRunner
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.recipe_processor import (
    RecipeData,
    RtExecutionInfo,
    pre_process_compiled,
)
from laboneq.controller.results import build_partial_result, make_acquired_result
from laboneq.controller.util import LabOneQControllerException, SweepParamsTracker
from laboneq.controller.versioning import SetupCaps, LabOneVersion
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.utilities.async_helpers import run_async
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements
from laboneq.data.execution_payload import TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import ScheduledExperiment

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse


_logger = logging.getLogger(__name__)


# Only recheck for the proper connected state if there was no check since more than
# the below amount of seconds. Important for performance with many small experiments
# executed in a batch.
CONNECT_CHECK_HOLDOFF = 10  # sec


class Controller:
    def __init__(
        self,
        target_setup: TargetSetup,
        ignore_version_mismatch: bool = False,
        neartime_callbacks: dict[str, Callable] | None = None,
    ):
        self._ignore_version_mismatch = ignore_version_mismatch
        self._do_emulation = True
        self._devices = DeviceCollection(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
        )

        self._setup_caps = SetupCaps(
            LabOneVersion.cast(
                zhinst_core_version(),
                raise_if_unsupported=not ignore_version_mismatch,
            )
        )

        self._last_connect_check_ts: float | None = None

        # Waves which are uploaded to the devices via pulse replacements
        self._current_waves = []
        self._neartime_callbacks: dict[str, Callable] = (
            {} if neartime_callbacks is None else neartime_callbacks
        )
        self._nodes_from_neartime_callbacks: dict[
            DeviceZI, NodeCollector
        ] = defaultdict(NodeCollector)
        self._recipe_data: RecipeData = None
        self._results = ExperimentResults()

        _logger.debug("Controller created")
        _logger.debug("Controller debug logging is on")

        _logger.info("VERSION: laboneq %s", __version__)

    @property
    def setup_caps(self) -> SetupCaps:
        return self._setup_caps

    def _allocate_resources(self):
        self._devices.free_allocations()
        osc_params = self._recipe_data.recipe.oscillator_params
        for osc_param in sorted(osc_params, key=lambda p: p.id):
            self._devices.find_by_uid(osc_param.device_id).allocate_osc(osc_param)

    async def _reset_to_idle_state(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self._devices.all:
                awaitables.append(device.collect_reset_nodes())

    async def _apply_recipe_initializations(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for initialization in self._recipe_data.initializations:
                device = self._devices.find_by_uid(initialization.device_uid)
                awaitables.append(
                    device.collect_initialization_nodes(
                        self._recipe_data.device_settings[initialization.device_uid],
                        initialization,
                        self._recipe_data,
                    )
                )
                awaitables.append(device.collect_osc_initialization_nodes())

    async def _set_nodes_before_awg_program_upload(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for initialization in self._recipe_data.initializations:
                device = self._devices.find_by_uid(initialization.device_uid)
                awaitables.append(
                    device.collect_awg_before_upload_nodes(
                        initialization, self._recipe_data
                    )
                )

    async def _perform_awg_upload(
        self,
        artifacts: list[
            tuple[
                DeviceZI, list[DaqNodeSetAction], list[DaqNodeSetAction], dict[str, Any]
            ]
        ],
    ):
        elf_upload_conditions: dict[DeviceZI, dict[str, Any]] = defaultdict(dict)
        elf_node_settings: list[DaqNodeSetAction] = []
        wf_node_settings: list[DaqNodeSetAction] = []

        for device, elf_nodes, wf_nodes, upload_ready_conditions in artifacts:
            elf_node_settings.extend(elf_nodes)
            wf_node_settings.extend(wf_nodes)
            if len(upload_ready_conditions) > 0:
                elf_upload_conditions[device].update(upload_ready_conditions)

        # Upload AWG programs, waveforms, and command tables:
        if len(elf_upload_conditions) > 0:
            for device in elf_upload_conditions.keys():
                await device.node_monitor.flush()

        _logger.debug("Started upload of AWG programs...")
        with tracing.get_tracer().start_span("upload-awg-programs") as _:
            await batch_set(elf_node_settings)

            if len(elf_upload_conditions) > 0:
                _logger.debug("Waiting for devices...")
                response_waiter = ResponseWaiter()
                for device, conditions in elf_upload_conditions.items():
                    response_waiter.add(target=device, conditions=conditions)
                timeout_s = 10
                if not await response_waiter.wait_all(timeout=timeout_s):
                    raise LabOneQControllerException(
                        f"AWGs not in ready state within timeout ({timeout_s} s). "
                        f"Not fulfilled:\n{response_waiter.remaining_str()}"
                    )
            if len(wf_node_settings) > 0:
                _logger.debug("Started upload of waveforms...")
                with tracing.get_tracer().start_span("upload-waveforms"):
                    await batch_set(wf_node_settings)
        _logger.debug("Finished upload.")

    async def _upload_awg_programs(self, nt_step: NtStepKey, rt_section_uid: str):
        # Mise en place:
        recipe_data = self._recipe_data

        async with gather_and_apply(self._perform_awg_upload) as awaitables:
            for initialization in recipe_data.initializations:
                if not initialization.awgs:
                    continue

                device = self._devices.find_by_uid(initialization.device_uid)

                for awg_obj in initialization.awgs:
                    awg_index = awg_obj.awg
                    awaitables.append(
                        device.prepare_artifacts(
                            recipe_data=recipe_data,
                            rt_section_uid=rt_section_uid,
                            initialization=initialization,
                            awg_index=awg_index,
                            nt_step=nt_step,
                        )
                    )

    async def _set_nodes_after_awg_program_upload(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for initialization in self._recipe_data.initializations:
                device = self._devices.find_by_uid(initialization.device_uid)
                awaitables.append(device.collect_awg_after_upload_nodes(initialization))

    async def _initialize_awgs(self, nt_step: NtStepKey, rt_section_uid: str):
        await self._set_nodes_before_awg_program_upload()
        await self._upload_awg_programs(nt_step=nt_step, rt_section_uid=rt_section_uid)
        await self._set_nodes_after_awg_program_upload()

    async def _configure_triggers(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for uid, device in itertools.chain(
                self._devices.leaders, self._devices.followers
            ):
                init = self._recipe_data.get_initialization_by_device_uid(uid)
                if init is None:
                    continue
                awaitables.append(
                    device.collect_trigger_configuration_nodes(init, self._recipe_data)
                )

    async def _prepare_nt_step(
        self, sweep_params_tracker: SweepParamsTracker
    ) -> list[DaqNodeSetAction]:
        for param in sweep_params_tracker.updated_params():
            self._recipe_data.attribute_value_tracker.update(
                param, sweep_params_tracker.get_param(param)
            )

        nt_sweep_nodes: list[DaqNodeSetAction] = []
        for device_uid, device in self._devices.all:
            nt_sweep_nodes.extend(
                await device.maybe_async(
                    device.collect_prepare_nt_step_nodes(
                        self._recipe_data.attribute_value_tracker.device_view(
                            device_uid
                        ),
                        self._recipe_data,
                    )
                )
            )

        self._recipe_data.attribute_value_tracker.reset_updated()

        return nt_sweep_nodes

    async def _initialize_devices(self):
        await self._reset_to_idle_state()
        self._allocate_resources()
        await self._apply_recipe_initializations()

    async def _execute_one_step_followers(self, with_pipeliner: bool):
        _logger.debug("Settings nodes to start on followers")

        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self._devices.followers:
                awaitables.append(
                    device.collect_execution_nodes(with_pipeliner=with_pipeliner)
                )

        response_waiter = ResponseWaiter()
        for _, device in self._devices.followers:
            response_waiter.add(
                target=device,
                conditions=await device.conditions_for_execution_ready(
                    with_pipeliner=with_pipeliner
                ),
            )
        if not await response_waiter.wait_all(timeout=2):
            _logger.warning(
                "Conditions to start RT on followers still not fulfilled after 2"
                " seconds, nonetheless trying to continue..."
                "\nNot fulfilled:\n%s",
                response_waiter.remaining_str(),
            )

        # Standalone workaround: The device is triggering itself,
        # thus split the execution into AWG trigger arming and triggering
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self._devices.followers:
                awaitables.append(device.collect_internal_start_execution_nodes())

    async def _execute_one_step_leaders(self, with_pipeliner: bool):
        _logger.debug("Settings nodes to start on leaders")
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self._devices.leaders:
                awaitables.append(
                    device.collect_execution_nodes(with_pipeliner=with_pipeliner)
                )

    async def _wait_execution_to_stop(
        self, acquisition_type: AcquisitionType, rt_execution_info: RtExecutionInfo
    ):
        min_wait_time = self._recipe_data.recipe.max_step_execution_time
        if rt_execution_info.with_pipeliner:
            pipeliner_reload_worst_case = 1500e-6
            min_wait_time = (
                min_wait_time + pipeliner_reload_worst_case
            ) * rt_execution_info.pipeliner_jobs
        if min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            _logger.info("Estimated RT execution time: %.2f s.", min_wait_time)
        guarded_wait_time = round(
            min_wait_time * 1.1 + 1
        )  # +10% and fixed 1sec guard time

        response_waiter = ResponseWaiter()
        for _, device in self._devices.followers:
            response_waiter.add(
                target=device,
                conditions=await device.conditions_for_execution_done(
                    acquisition_type, with_pipeliner=rt_execution_info.with_pipeliner
                ),
            )
        if not await response_waiter.wait_all(timeout=guarded_wait_time):
            _logger.warning(
                (
                    "Stop conditions still not fulfilled after %f s, estimated"
                    " execution time was %.2f s. Continuing to the next step."
                    "\nNot fulfilled:\n%s"
                ),
                guarded_wait_time,
                min_wait_time,
                response_waiter.remaining_str(),
            )

    async def _setup_one_step_execution(self, with_pipeliner: bool):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for device_uid, device in self._devices.all:
                has_awg_in_use = any(
                    init.device_uid == device_uid and len(init.awgs) > 0
                    for init in self._recipe_data.initializations
                )
                awaitables.append(
                    device.collect_execution_setup_nodes(
                        with_pipeliner=with_pipeliner, has_awg_in_use=has_awg_in_use
                    )
                )

    async def _teardown_one_step_execution(self, with_pipeliner: bool):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self._devices.all:
                awaitables.append(
                    device.collect_execution_teardown_nodes(
                        with_pipeliner=with_pipeliner
                    )
                )

    async def _execute_one_step(
        self, acquisition_type: AcquisitionType, rt_section_uid: str
    ):
        _logger.debug("Step executing")

        await self._devices.flush_monitor()

        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]

        await self._setup_one_step_execution(
            with_pipeliner=rt_execution_info.with_pipeliner
        )

        # Can't batch everything together, because PQSC needs to be executed after HDs
        # otherwise it can finish before AWGs are started, and the trigger is lost
        await self._execute_one_step_followers(
            with_pipeliner=rt_execution_info.with_pipeliner
        )
        await self._execute_one_step_leaders(
            with_pipeliner=rt_execution_info.with_pipeliner
        )

        _logger.debug("Execution started")

        await self._wait_execution_to_stop(
            acquisition_type, rt_execution_info=rt_execution_info
        )
        await self._teardown_one_step_execution(
            with_pipeliner=rt_execution_info.with_pipeliner
        )

        _logger.debug("Execution stopped")

    def connect(self, do_emulation: bool = True, reset_devices: bool = False):
        self._do_emulation = (
            do_emulation  # Remember mode for later implicit connect check
        )
        run_async(self._connect_async, reset_devices=reset_devices)

    async def _connect_async(self, reset_devices: bool = False):
        now = time.monotonic()
        if (
            self._last_connect_check_ts is None
            or now - self._last_connect_check_ts > CONNECT_CHECK_HOLDOFF
        ):
            await self._devices.connect(
                do_emulation=self._do_emulation, reset_devices=reset_devices
            )
        self._last_connect_check_ts = now

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        run_async(
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
        run_async(self._disconnect_async)

    async def _disconnect_async(self):
        _logger.info("Disconnecting from all devices and servers...")
        await self._devices.disconnect()
        self._last_connect_check_ts = None
        _logger.info("Successfully disconnected from all devices and servers.")

    def execute_compiled(
        self,
        scheduled_experiment: ScheduledExperiment,
        protected_session: ProtectedSession | None = None,
    ) -> ExperimentResults:
        return run_async(
            self._execute_compiled_async, scheduled_experiment, protected_session
        )

    async def _execute_compiled_async(
        self,
        scheduled_experiment: ScheduledExperiment,
        protected_session: ProtectedSession | None = None,
    ) -> ExperimentResults:
        self._recipe_data = pre_process_compiled(
            scheduled_experiment,
            self._devices,
            scheduled_experiment.execution,
            self._setup_caps,
        )
        return await self._execute_compiled_impl(
            protected_session=protected_session or ProtectedSession(None)
        )

    async def _execute_compiled_impl(
        self, protected_session: ProtectedSession
    ) -> ExperimentResults:
        await (
            self._connect_async()
        )  # Ensure all connect configurations are still valid!
        self._prepare_result_shapes()
        protected_session._set_experiment_results(self._results)
        try:
            await self._initialize_devices()

            # Ensure no side effects from the previous execution in the same session
            self._current_waves = []
            self._nodes_from_neartime_callbacks.clear()
            _logger.info("Starting near-time execution...")
            try:
                with tracing.get_tracer().start_span("near-time-execution"):
                    await NearTimeRunner(
                        controller=self,
                        protected_session=protected_session,
                    ).run(self._recipe_data.execution)
            except AbortExecution:
                # eat the exception
                pass
            _logger.info("Finished near-time execution.")
            await self._devices.check_errors()
        except LabOneQControllerException:
            # Report device errors if any - it maybe useful to diagnose the original exception
            device_errors = await self._devices.check_errors(raise_on_error=False)
            if device_errors is not None:
                _logger.warning(device_errors)
            raise
        finally:
            # Ensure that the experiment run time is not included in the idle timeout for the connection check.
            self._last_connect_check_ts = time.monotonic()

        await self._devices.on_experiment_end()

        return self._results

    def _find_awg(self, seqc_name: str) -> tuple[str, int]:
        # TODO(2K): Do this in the recipe preprocessor, or even modify the compiled experiment
        #  data model
        for rt_exec_step in self._recipe_data.recipe.realtime_execution_init:
            if rt_exec_step.seqc_ref == seqc_name:
                return rt_exec_step.device_id, rt_exec_step.awg_id
        return None, None

    def replace_pulse(
        self, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        """Replaces specific pulse with the new sample data on the device.

        This is useful when called from the near-time callback, allows fast
        waveform replacement within near-time loop without experiment
        recompilation.

        Args:
            pulse_uid: pulse to replace, can be Pulse object or uid of the pulse
            pulse_or_array: replacement pulse, can be Pulse object or value array
            (see sampled_pulse_* from the pulse library)
        """
        if isinstance(pulse_uid, str):
            for waveform in self._recipe_data.scheduled_experiment.pulse_map[
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
                f"Pulse {pulse_uid.uid}"
            )

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            self._recipe_data.rt_execution_infos
        )
        wave_replacements = calc_wave_replacements(
            self._recipe_data.scheduled_experiment,
            pulse_uid,
            pulse_or_array,
            self._current_waves,
        )
        for repl in wave_replacements:
            awg_indices = next(
                a
                for a in self._recipe_data.scheduled_experiment.wave_indices
                if a["filename"] == repl.awg_id
            )
            awg_wave_map: dict[str, list[int | str]] = awg_indices["value"]
            target_wave = awg_wave_map.get(repl.sig_string)
            seqc_name = repl.awg_id
            awg = self._find_awg(seqc_name)
            device = self._devices.find_by_uid(awg[0])

            if repl.replacement_type == ReplacementType.I_Q:
                clipped = np.clip(repl.samples, -1.0, 1.0)
                bin_wave = zhinst.utils.convert_awg_waveform(*clipped)
                self._nodes_from_neartime_callbacks[device].extend(
                    device.prepare_upload_binary_wave(
                        filename=repl.sig_string + " (repl)",
                        waveform=bin_wave,
                        awg_index=awg[1],
                        wave_index=target_wave[0],
                        acquisition_type=acquisition_type,
                    )
                )
            elif repl.replacement_type == ReplacementType.COMPLEX:
                np.clip(repl.samples.real, -1.0, 1.0, out=repl.samples.real)
                np.clip(repl.samples.imag, -1.0, 1.0, out=repl.samples.imag)
                self._nodes_from_neartime_callbacks[device].extend(
                    device.prepare_upload_binary_wave(
                        filename=repl.sig_string + " (repl)",
                        waveform=repl.samples,
                        awg_index=awg[1],
                        wave_index=target_wave[0],
                        acquisition_type=acquisition_type,
                    )
                )

    async def _prepare_rt_execution(
        self, rt_section_uid: str
    ) -> list[DaqNodeSetAction]:
        if rt_section_uid is None:
            return [], []  # Old recipe-based execution - skip RT preparation
        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]

        nodes_to_prepare_rt = []
        for device, nc in self._nodes_from_neartime_callbacks.items():
            nodes_to_prepare_rt.extend(await device.maybe_async(nc))
        self._nodes_from_neartime_callbacks.clear()

        for _, device in self._devices.leaders:
            nodes_to_prepare_rt.extend(
                await device.configure_feedback(self._recipe_data)
            )
        for awg_key, awg_config in self._recipe_data.awgs_producing_results():
            device = self._devices.find_by_uid(awg_key.device_uid)
            if rt_execution_info.averaging_mode == AveragingMode.SINGLE_SHOT:
                effective_averages = 1
                effective_averaging_mode = AveragingMode.CYCLIC
                # TODO(2K): handle sequential
            else:
                effective_averages = rt_execution_info.averages
                effective_averaging_mode = rt_execution_info.averaging_mode
            nodes_to_prepare_rt.extend(
                await device.configure_acquisition(
                    awg_key,
                    awg_config,
                    self._recipe_data.recipe.integrator_allocations,
                    effective_averages,
                    effective_averaging_mode,
                    rt_execution_info.acquisition_type,
                    rt_execution_info.with_pipeliner,
                )
            )
        return nodes_to_prepare_rt

    def _prepare_result_shapes(self):
        self._results = ExperimentResults()
        if len(self._recipe_data.rt_execution_infos) == 0:
            return
        if len(self._recipe_data.rt_execution_infos) > 1:
            raise LabOneQControllerException(
                "Multiple 'acquire_loop_rt' sections per experiment is not supported."
            )
        rt_info = next(iter(self._recipe_data.rt_execution_infos.values()))
        for handle, shape_info in self._recipe_data.result_shapes.items():
            if rt_info.acquisition_type == AcquisitionType.RAW:
                signal_id = rt_info.signal_by_handle(handle)
                awg_config = self._recipe_data.awg_config_by_acquire_signal(signal_id)
                # Use default length 4096, in case AWG config is not available
                raw_acquire_length = (
                    4096 if awg_config is None else awg_config.raw_acquire_length
                )
                # TODO: This result format does not work when sweeping in near-time, returns only
                # results of the last sweep parameter value
                empty_res = make_acquired_result(
                    data=np.empty(shape=[raw_acquire_length], dtype=np.complex128),
                    axis_name=["samples"],
                    axis=[np.arange(raw_acquire_length)],
                    handle=handle,
                )
                empty_res.data[:] = np.nan
                self._results.acquired_results[handle] = empty_res
            else:
                axis_name = deepcopy(shape_info.base_axis_name)
                axis = deepcopy(shape_info.base_axis)
                shape = deepcopy(shape_info.base_shape)
                if shape_info.additional_axis > 1:
                    axis_name.append(handle)
                    axis.append(
                        np.linspace(
                            0,
                            shape_info.additional_axis - 1,
                            shape_info.additional_axis,
                        )
                    )
                    shape.append(shape_info.additional_axis)
                empty_res = make_acquired_result(
                    data=np.empty(shape=tuple(shape), dtype=np.complex128),
                    axis_name=axis_name,
                    axis=axis,
                    handle=handle,
                )
                if len(shape) == 0:
                    empty_res.data = np.nan
                else:
                    empty_res.data[:] = np.nan
                self._results.acquired_results[handle] = empty_res

    async def _read_one_step_results(self, nt_step: NtStepKey, rt_section_uid: str):
        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]
        for awg_key, awg_config in self._recipe_data.awgs_producing_results():
            device = self._devices.find_by_uid(awg_key.device_uid)
            if rt_execution_info.acquisition_type == AcquisitionType.RAW:
                raw_results = await device.get_input_monitor_data(
                    awg_key.awg_index, awg_config.raw_acquire_length
                )
                # Copy to all result handles, but actually only one handle is supported for now
                for signal in awg_config.acquire_signals:
                    mapping = rt_execution_info.signal_result_map.get(signal, [])
                    unique_handles = set(mapping)
                    for handle in unique_handles:
                        if handle is None:
                            continue  # Ignore unused acquire signal if any
                        result = self._results.acquired_results[handle]
                        for raw_result_idx, raw_result in enumerate(raw_results):
                            result.data[raw_result_idx] = raw_result
            else:
                if rt_execution_info.averaging_mode == AveragingMode.SINGLE_SHOT:
                    effective_averages = 1
                else:
                    effective_averages = rt_execution_info.averages
                await device.check_results_acquired_status(
                    awg_key.awg_index,
                    rt_execution_info.acquisition_type,
                    awg_config.result_length,
                    effective_averages,
                )
                for signal in awg_config.acquire_signals:
                    integrator_allocation = next(
                        (
                            i
                            for i in self._recipe_data.recipe.integrator_allocations
                            if (
                                i.signal_id
                                if isinstance(i.signal_id, str)
                                else i.signal_id[0]
                            )
                            == signal
                        ),
                        None,
                    )
                    if not integrator_allocation:
                        continue
                    is_multistate = not isinstance(integrator_allocation.signal_id, str)
                    assert integrator_allocation.device_id == awg_key.device_uid
                    assert integrator_allocation.awg == awg_key.awg_index
                    result_indices = (
                        integrator_allocation.channels[0]
                        if is_multistate
                        else integrator_allocation.channels
                    )
                    raw_results = await device.get_measurement_data(
                        self._recipe_data,
                        awg_key.awg_index,
                        rt_execution_info,
                        result_indices,
                        awg_config.result_length,
                        effective_averages,
                    )
                    mapping = rt_execution_info.signal_result_map.get(signal, [])
                    unique_handles = set(mapping)
                    for handle in unique_handles:
                        if handle is None:
                            continue  # unused entries in sparse result vector map to None handle
                        result = self._results.acquired_results[handle]
                        build_partial_result(
                            result, nt_step, raw_results, mapping, handle
                        )

    def _report_step_error(self, nt_step: NtStepKey, rt_section_uid: str, message: str):
        self._results.execution_errors.append(
            (list(nt_step.indices), rt_section_uid, message)
        )
