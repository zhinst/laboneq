# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import os
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import zhinst.utils
from numpy import typing as npt

from laboneq import __version__
from laboneq._observability import tracing
from laboneq.controller.communication import (
    DaqNodeAction,
    DaqNodeSetAction,
    DaqWrapper,
    batch_set,
)
from laboneq.controller.devices.device_collection import DeviceCollection
from laboneq.controller.devices.device_uhfqa import DeviceUHFQA
from laboneq.controller.devices.device_zi import Waveforms
from laboneq.controller.devices.zi_node_monitor import ResponseWaiter
from laboneq.controller.near_time_runner import NearTimeRunner
from laboneq.controller.pipeliner_reload_tracker import PipelinerReloadTracker
from laboneq.controller.recipe_processor import (
    AwgKey,
    RecipeData,
    RtExecutionInfo,
    pre_process_compiled,
)
from laboneq.controller.results import build_partial_result, make_acquired_result
from laboneq.controller.util import LabOneQControllerException, SweepParamsTracker
from laboneq.controller.versioning import LabOneVersion
from laboneq.core.exceptions import AbortExecution
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.utilities.async_helpers import run_async
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements
from laboneq.data.execution_payload import TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.recipe import NtStepKey
from laboneq.executor.execution_from_experiment import ExecutionFactoryFromExperiment
from laboneq.executor.executor import Statement

if TYPE_CHECKING:
    from laboneq.controller.devices.device_zi import DeviceZI
    from laboneq.core.types import CompiledExperiment
    from laboneq.data.execution_payload import ExecutionPayload
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.session import Session


_logger = logging.getLogger(__name__)


# Only recheck for the proper connected state if there was no check since more than
# the below amount of seconds. Important for performance with many small experiments
# executed in a batch.
CONNECT_CHECK_HOLDOFF = 10  # sec


class ControllerRunParameters:
    shut_down: bool = False
    dry_run: bool = False
    disconnect: bool = False
    working_dir: str = "laboneq_output"
    setup_filename = None
    servers_filename = None
    ignore_version_mismatch = False
    reset_devices = False


# atexit hook
def _stop_controller(controller: "Controller"):
    controller.shut_down()


@dataclass
class _SeqCCompileItem:
    awg_index: int
    seqc_code: str | None = None
    seqc_filename: str | None = None
    elf: bytes | None = None


@dataclass
class _UploadItem:
    seqc_item: _SeqCCompileItem | None
    waves: Waveforms | None
    command_table: dict[Any] | None


class Controller:
    def __init__(
        self,
        run_parameters: ControllerRunParameters | None = None,
        target_setup: TargetSetup | None = None,
        neartime_callbacks: dict[str, Callable] | None = None,
    ):
        self._run_parameters = run_parameters or ControllerRunParameters()
        self._devices = DeviceCollection(
            target_setup,
            self._run_parameters.dry_run,
            self._run_parameters.ignore_version_mismatch,
            self._run_parameters.reset_devices,
        )

        self._dataserver_version: LabOneVersion | None = None

        self._last_connect_check_ts: float = None

        # Waves which are uploaded to the devices via pulse replacements
        self._current_waves = []
        self._neartime_callbacks: dict[str, Callable] = neartime_callbacks
        self._nodes_from_neartime_callbacks: list[DaqNodeAction] = []
        self._recipe_data: RecipeData = None
        self._session: Any = None
        self._results = ExperimentResults()
        self._pipeliner_reload_tracker = PipelinerReloadTracker()

        _logger.debug("Controller created")
        _logger.debug("Controller debug logging is on")

        _logger.info("VERSION: laboneq %s", __version__)

    def _allocate_resources(self):
        self._devices.free_allocations()
        osc_params = self._recipe_data.recipe.oscillator_params
        for osc_param in sorted(osc_params, key=lambda p: p.id):
            self._devices.find_by_uid(osc_param.device_id).allocate_osc(osc_param)

    async def _reset_to_idle_state(self):
        reset_nodes = []
        for _, device in self._devices.all:
            reset_nodes.extend(device.collect_reset_nodes())
        await batch_set(reset_nodes)

    async def _apply_recipe_initializations(self):
        nodes_to_initialize: list[DaqNodeAction] = []
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_initialization_nodes(
                    self._recipe_data.device_settings[initialization.device_uid],
                    initialization,
                    self._recipe_data,
                )
            )
            nodes_to_initialize.extend(device.collect_osc_initialization_nodes())

        await batch_set(nodes_to_initialize)

    async def _set_nodes_before_awg_program_upload(self):
        nodes_to_initialize = []
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_awg_before_upload_nodes(
                    initialization, self._recipe_data
                )
            )
        await batch_set(nodes_to_initialize)

    @tracing.trace("awg-program-handler")
    async def _upload_awg_programs(self, nt_step: NtStepKey, rt_section_uid: str):
        # Mise en place:
        awg_data: dict[DeviceZI, list[_UploadItem]] = defaultdict(list)
        awgs_used: dict[DeviceZI, set[int]] = defaultdict(set)
        compile_data: dict[DeviceZI, list[_SeqCCompileItem]] = defaultdict(list)
        recipe_data = self._recipe_data
        rt_execution_info = recipe_data.rt_execution_infos.get(rt_section_uid)
        with_pipeliner = rt_execution_info.pipeliner_chunk_count is not None
        acquisition_type = RtExecutionInfo.get_acquisition_type_def(rt_execution_info)
        for initialization in recipe_data.initializations:
            if not initialization.awgs:
                continue

            device = self._devices.find_by_uid(initialization.device_uid)
            if with_pipeliner and not device.has_pipeliner:
                raise LabOneQControllerException(
                    f"{device.dev_repr}: Pipeliner is not supported by the device."
                )

            for awg_obj in initialization.awgs:
                awg_index = awg_obj.awg
                awgs_used[device].add(awg_index)
                for pipeline_chunk in range(
                    rt_execution_info.pipeliner_chunk_count or 1
                ):
                    effective_nt_step = (
                        NtStepKey(indices=tuple([*nt_step.indices, pipeline_chunk]))
                        if with_pipeliner
                        else nt_step
                    )
                    rt_exec_step = next(
                        (
                            r
                            for r in recipe_data.recipe.realtime_execution_init
                            if r.device_id == initialization.device_uid
                            and r.awg_id == awg_obj.awg
                            and r.nt_step == effective_nt_step
                        ),
                        None,
                    )

                    if rt_execution_info.pipeliner_chunk_count is None:
                        seqc_filename = (
                            None if rt_exec_step is None else rt_exec_step.seqc_ref
                        )
                    else:
                        # TODO(2K): repeated compilation of SeqC to be solved by moving it to the compile stage
                        (
                            rt_exec_step,
                            seqc_filename,
                        ) = self._pipeliner_reload_tracker.calc_next_step(
                            awg_key=AwgKey(
                                device_uid=initialization.device_uid,
                                awg_index=awg_index,
                            ),
                            pipeline_chunk=pipeline_chunk,
                            rt_exec_step=rt_exec_step,
                        )

                    if rt_exec_step is None:
                        continue

                    seqc_code = device.prepare_seqc(
                        recipe_data.scheduled_experiment.artifacts,
                        rt_exec_step.seqc_ref,
                    )
                    waves = device.prepare_waves(
                        recipe_data.scheduled_experiment.artifacts,
                        rt_exec_step.wave_indices_ref,
                    )
                    command_table = device.prepare_command_table(
                        recipe_data.scheduled_experiment.artifacts,
                        rt_exec_step.wave_indices_ref,
                    )

                    seqc_item = _SeqCCompileItem(
                        awg_index=awg_index,
                    )

                    if seqc_code is not None:
                        seqc_item.seqc_code = seqc_code
                        seqc_item.seqc_filename = seqc_filename
                        compile_data[device].append(seqc_item)

                    awg_data[device].append(
                        _UploadItem(
                            seqc_item=seqc_item,
                            waves=waves,
                            command_table=command_table,
                        )
                    )

        if compile_data:
            self._awg_compile(compile_data)

        # Upload AWG programs, waveforms, and command tables:
        elf_node_settings: dict[DaqWrapper, list[DaqNodeSetAction]] = defaultdict(list)
        elf_upload_conditions: dict[DaqWrapper, dict[str, Any]] = defaultdict(dict)
        wf_node_settings: dict[DaqWrapper, list[DaqNodeSetAction]] = defaultdict(list)
        for device, items in awg_data.items():
            if with_pipeliner:
                for awg_index in awgs_used[device]:
                    elf_node_settings[device.daq].extend(
                        device.pipeliner_prepare_for_upload(awg_index)
                    )
            for item in items:
                seqc_item = item.seqc_item
                if seqc_item.elf is not None:
                    set_action = device.prepare_upload_elf(
                        seqc_item.elf, seqc_item.awg_index, seqc_item.seqc_filename
                    )
                    node_settings = elf_node_settings[device.daq]
                    node_settings.append(set_action)

                    if isinstance(device, DeviceUHFQA):
                        # UHFQA does not yet support upload of ELF and waveforms in
                        # a single transaction.
                        ready_node = device.get_sequencer_paths(
                            seqc_item.awg_index
                        ).ready
                        elf_upload_conditions[device.daq][ready_node] = 1

                if isinstance(device, DeviceUHFQA):
                    wf_dev_nodes = wf_node_settings[device.daq]
                else:
                    wf_dev_nodes = elf_node_settings[device.daq]

                if item.waves is not None:
                    wf_dev_nodes += device.prepare_upload_all_binary_waves(
                        seqc_item.awg_index, item.waves, acquisition_type
                    )

                if item.command_table is not None:
                    set_action = device.prepare_upload_command_table(
                        seqc_item.awg_index, item.command_table
                    )
                    wf_dev_nodes.append(set_action)

                if with_pipeliner:
                    # For devices with pipeliner, wf_dev_nodes == elf_node_settings
                    wf_dev_nodes.extend(device.pipeliner_commit(seqc_item.awg_index))

            if with_pipeliner:
                for awg_index in awgs_used[device]:
                    elf_upload_conditions[device.daq].update(
                        device.pipeliner_ready_conditions(awg_index)
                    )

        if len(elf_upload_conditions) > 0:
            for daq in elf_upload_conditions.keys():
                daq.node_monitor.flush()

        _logger.debug("Started upload of AWG programs...")
        with tracing.get_tracer().start_span("upload-awg-programs") as _:
            for daq, nodes in elf_node_settings.items():
                await daq.batch_set(nodes)

            if len(elf_upload_conditions) > 0:
                _logger.debug("Waiting for devices...")
                response_waiter = ResponseWaiter()
                for daq, conditions in elf_upload_conditions.items():
                    response_waiter.add(
                        target=daq.node_monitor,
                        conditions=conditions,
                    )
                timeout_s = 10
                if not response_waiter.wait_all(timeout=timeout_s):
                    raise LabOneQControllerException(
                        f"AWGs not in ready state within timeout ({timeout_s} s). "
                        f"Not fulfilled:\n{response_waiter.remaining_str()}"
                    )
            if len(wf_node_settings) > 0:
                _logger.debug("Started upload of waveforms...")
                with tracing.get_tracer().start_span("upload-waveforms") as _:
                    for daq, nodes in wf_node_settings.items():
                        await daq.batch_set(nodes)
        _logger.debug("Finished upload.")

    @classmethod
    def _awg_compile(cls, awg_data: dict[DeviceZI, list[_SeqCCompileItem]]):
        # Compile in parallel:
        def worker(device: DeviceZI, item: _SeqCCompileItem, span: tracing.Span):
            with tracing.get_tracer().start_span("compile-awg-thread", span) as _:
                item.elf = device.compile_seqc(
                    item.seqc_code, item.awg_index, item.seqc_filename
                )

        _logger.debug("Started compilation of AWG programs...")
        with tracing.get_tracer().start_span("compile-awg-programs") as awg_span:
            max_workers = os.environ.get("LABONEQ_AWG_COMPILER_MAX_WORKERS")
            max_workers = int(max_workers) if max_workers is not None else None
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(worker, device, item, awg_span)
                    for device, items in awg_data.items()
                    for item in items
                ]
                concurrent.futures.wait(futures)
                exceptions = [
                    future.exception()
                    for future in futures
                    if future.exception() is not None
                ]
                if len(exceptions) > 0:
                    raise LabOneQControllerException(
                        "Compilation failed. See log output for details."
                    )
        _logger.debug("Finished compilation.")

    async def _set_nodes_after_awg_program_upload(self):
        nodes_to_initialize = []
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_awg_after_upload_nodes(initialization)
            )

        await batch_set(nodes_to_initialize)

    async def _initialize_awgs(self, nt_step: NtStepKey, rt_section_uid: str):
        await self._set_nodes_before_awg_program_upload()
        await self._upload_awg_programs(nt_step=nt_step, rt_section_uid=rt_section_uid)
        await self._set_nodes_after_awg_program_upload()

    async def _configure_triggers(self):
        nodes_to_configure_triggers = []

        for uid, device in itertools.chain(
            self._devices.leaders, self._devices.followers
        ):
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            nodes_to_configure_triggers.extend(
                await device.collect_trigger_configuration_nodes(
                    init, self._recipe_data
                )
            )

        await batch_set(nodes_to_configure_triggers)

    def _prepare_nt_step(
        self, sweep_params_tracker: SweepParamsTracker
    ) -> list[DaqNodeAction]:
        for param in sweep_params_tracker.updated_params():
            self._recipe_data.attribute_value_tracker.update(
                param, sweep_params_tracker.get_param(param)
            )

        nt_sweep_nodes: list[DaqNodeAction] = []
        for device_uid, device in self._devices.all:
            nt_sweep_nodes.extend(
                device.collect_prepare_nt_step_nodes(
                    self._recipe_data.attribute_value_tracker.device_view(device_uid),
                    self._recipe_data,
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

        nodes_to_execute = []
        for _, device in self._devices.followers:
            nodes_to_execute.extend(
                device.collect_execution_nodes(with_pipeliner=with_pipeliner)
            )

        await batch_set(nodes_to_execute)

        response_waiter = ResponseWaiter()
        for _, device in self._devices.followers:
            response_waiter.add(
                target=device.daq.node_monitor,
                conditions=device.conditions_for_execution_ready(
                    with_pipeliner=with_pipeliner
                ),
            )
        if not response_waiter.wait_all(timeout=2):
            _logger.warning(
                "Conditions to start RT on followers still not fulfilled after 2"
                " seconds, nonetheless trying to continue..."
                "\nNot fulfilled:\n%s",
                response_waiter.remaining_str(),
            )

        # Standalone workaround: The device is triggering itself,
        # thus split the execution into AWG trigger arming and triggering
        nodes_to_execute = []
        for _, device in self._devices.followers:
            nodes_to_execute.extend(device.collect_internal_start_execution_nodes())

        await batch_set(nodes_to_execute)

    async def _execute_one_step_leaders(self, with_pipeliner: bool):
        _logger.debug("Settings nodes to start on leaders")
        nodes_to_execute = []

        for _, device in self._devices.leaders:
            nodes_to_execute.extend(
                device.collect_execution_nodes(with_pipeliner=with_pipeliner)
            )
        await batch_set(nodes_to_execute)

    def _wait_execution_to_stop(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ):
        min_wait_time = self._recipe_data.recipe.max_step_execution_time
        if min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            _logger.info("Estimated RT execution time: %.2f s.", min_wait_time)
        guarded_wait_time = round(
            min_wait_time * 1.1 + 1
        )  # +10% and fixed 1sec guard time

        response_waiter = ResponseWaiter()
        for _, device in self._devices.followers:
            response_waiter.add(
                target=device.daq.node_monitor,
                conditions=device.conditions_for_execution_done(
                    acquisition_type, with_pipeliner=with_pipeliner
                ),
            )
        if not response_waiter.wait_all(timeout=guarded_wait_time):
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
        nodes_to_execute = []
        for device_uid, device in self._devices.all:
            has_awg_in_use = any(
                init.device_uid == device_uid and len(init.awgs) > 0
                for init in self._recipe_data.initializations
            )
            nodes_to_execute.extend(
                device.collect_execution_setup_nodes(
                    with_pipeliner=with_pipeliner, has_awg_in_use=has_awg_in_use
                )
            )
        await batch_set(nodes_to_execute)

    async def _teardown_one_step_execution(self, with_pipeliner: bool):
        nodes_to_execute = []
        for _, device in self._devices.all:
            nodes_to_execute.extend(
                device.collect_execution_teardown_nodes(with_pipeliner=with_pipeliner)
            )
        await batch_set(nodes_to_execute)

    async def _execute_one_step(
        self, acquisition_type: AcquisitionType, rt_section_uid: str
    ):
        _logger.debug("Step executing")

        self._devices.flush_monitor()

        rt_execution_info = self._recipe_data.rt_execution_infos.get(rt_section_uid)
        with_pipeliner = rt_execution_info.pipeliner_chunk_count is not None

        await self._setup_one_step_execution(with_pipeliner=with_pipeliner)

        # Can't batch everything together, because PQSC needs to be executed after HDs
        # otherwise it can finish before AWGs are started, and the trigger is lost
        await self._execute_one_step_followers(with_pipeliner=with_pipeliner)
        await self._execute_one_step_leaders(with_pipeliner=with_pipeliner)

        _logger.debug("Execution started")

        self._wait_execution_to_stop(acquisition_type, with_pipeliner=with_pipeliner)
        await self._teardown_one_step_execution(with_pipeliner=with_pipeliner)

        _logger.debug("Execution stopped")

    def connect(self):
        run_async(self.connect_async())

    async def connect_async(self):
        now = time.monotonic()
        if (
            self._last_connect_check_ts is None
            or now - self._last_connect_check_ts > CONNECT_CHECK_HOLDOFF
        ):
            await self._devices.connect()

        try:
            self._dataserver_version = next(self._devices.leaders)[
                1
            ].daq._dataserver_version
        except StopIteration:
            # It may happen in emulation mode, mainly for tests
            # We use LATEST in emulation mode, keeping the consistency here.
            self._dataserver_version = LabOneVersion.LATEST

        self._last_connect_check_ts = now

    def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        run_async(self.disable_outputs_async(device_uids, logical_signals, unused_only))

    async def disable_outputs_async(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        await self._devices.disable_outputs(device_uids, logical_signals, unused_only)

    def shut_down(self):
        run_async(self.shut_down_async())

    async def shut_down_async(self):
        _logger.info("Shutting down all devices...")
        self._devices.shut_down()
        _logger.info("Successfully Shut down all devices.")

    def disconnect(self):
        run_async(self.disconnect_async())

    async def disconnect_async(self):
        _logger.info("Disconnecting from all devices and servers...")
        self._devices.disconnect()
        self._last_connect_check_ts = None
        _logger.info("Successfully disconnected from all devices and servers.")

    # TODO(2K): remove legacy code
    def execute_compiled_legacy(
        self, compiled_experiment: CompiledExperiment, session: Session | None = None
    ):
        run_async(self.execute_compiled_legacy_async(compiled_experiment, session))

    async def execute_compiled_legacy_async(
        self, compiled_experiment: CompiledExperiment, session: Session | None = None
    ):
        execution: Statement
        if hasattr(compiled_experiment.scheduled_experiment, "execution"):
            execution = compiled_experiment.scheduled_experiment.execution
        else:
            execution = ExecutionFactoryFromExperiment().make(
                compiled_experiment.experiment
            )

        self._recipe_data = pre_process_compiled(
            compiled_experiment.scheduled_experiment,
            self._devices,
            execution,
        )

        self._session = session
        await self._execute_compiled_impl()
        if session and session._last_results:
            session._last_results.acquired_results = self._results.acquired_results
            session._last_results.neartime_callback_results = (
                self._results.neartime_callback_results
            )
            session._last_results.execution_errors = self._results.execution_errors

    def execute_compiled(self, job: ExecutionPayload):
        run_async(self.execute_compiled_async(job))

    async def execute_compiled_async(self, job: ExecutionPayload):
        self._recipe_data = pre_process_compiled(
            job.scheduled_experiment,
            self._devices,
            job.scheduled_experiment.execution,
        )
        self._session = None
        await self._execute_compiled_impl()

    async def _execute_compiled_impl(self):
        await self.connect_async()  # Ensure all connect configurations are still valid!
        self._prepare_result_shapes()
        try:
            await self._initialize_devices()

            # Ensure no side effects from the previous execution in the same session
            self._current_waves = []
            self._nodes_from_neartime_callbacks = []
            _logger.info("Starting near-time execution...")
            try:
                with tracing.get_tracer().start_span("near-time-execution"):
                    await NearTimeRunner(controller=self).run(
                        self._recipe_data.execution
                    )
            except AbortExecution:
                # eat the exception
                pass
            _logger.info("Finished near-time execution.")
            self._devices.check_errors()
        except LabOneQControllerException:
            # Report device errors if any - it maybe useful to diagnose the original exception
            device_errors = self._devices.check_errors(raise_on_error=False)
            if device_errors is not None:
                _logger.warning(device_errors)
            raise
        finally:
            # Ensure that the experiment run time is not included in the idle timeout for the connection check.
            self._last_connect_check_ts = time.monotonic()

        await self._devices.on_experiment_end()

        if self._run_parameters.shut_down is True:
            await self.shut_down_async()

        if self._run_parameters.disconnect is True:
            await self.disconnect_async()

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
                self._nodes_from_neartime_callbacks.append(
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
                self._nodes_from_neartime_callbacks.append(
                    device.prepare_upload_binary_wave(
                        filename=repl.sig_string + " (repl)",
                        waveform=repl.samples,
                        awg_index=awg[1],
                        wave_index=target_wave[0],
                        acquisition_type=acquisition_type,
                    )
                )

    def _prepare_rt_execution(self, rt_section_uid: str) -> list[DaqNodeAction]:
        if rt_section_uid is None:
            return [], []  # Old recipe-based execution - skip RT preparation
        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]
        self._nodes_from_neartime_callbacks.sort(key=lambda v: v.path)
        nodes_to_prepare_rt = [*self._nodes_from_neartime_callbacks]
        self._nodes_from_neartime_callbacks.clear()
        for _, device in self._devices.leaders:
            nodes_to_prepare_rt.extend(device.configure_feedback(self._recipe_data))
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
                device.configure_acquisition(
                    awg_key,
                    awg_config,
                    self._recipe_data.recipe.integrator_allocations,
                    effective_averages,
                    effective_averaging_mode,
                    rt_execution_info.acquisition_type,
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
                raw_results = device.get_input_monitor_data(
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
                    raw_results = device.get_measurement_data(
                        awg_key.awg_index,
                        rt_execution_info.acquisition_type,
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
