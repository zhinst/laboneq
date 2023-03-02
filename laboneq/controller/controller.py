# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import concurrent.futures
import itertools
import logging
import os
import traceback
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import zhinst.utils
from numpy import typing as npt

from laboneq import __version__
from laboneq._observability import tracing
from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
    DaqWrapper,
    batch_set,
)
from laboneq.controller.devices.device_collection import DeviceCollection
from laboneq.controller.devices.device_uhfqa import DeviceUHFQA
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.devices.zi_node_monitor import ResponseWaiter
from laboneq.controller.protected_session import ProtectedSession
from laboneq.controller.recipe_1_4_0 import *  # noqa: F401, F403
from laboneq.controller.recipe_processor import (
    RecipeData,
    RtExecutionInfo,
    pre_process_compiled,
)
from laboneq.controller.results import (
    build_partial_result,
    make_acquired_result,
    make_empty_results,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements
from laboneq.executor.executor import ExecutorBase, LoopingMode

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl import Session
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.result.results import Results


class ControllerRunParameters:
    shut_down: bool = False
    dry_run: bool = False
    disconnect: bool = False
    working_dir: str = "laboneq_output"
    setup_filename = None
    servers_filename = None
    ignore_lab_one_version_error = False


# atexit hook
def _stop_controller(controller: "Controller"):
    controller.shut_down()


class Controller:
    def __init__(
        self,
        run_parameters: ControllerRunParameters = None,
        device_setup: DeviceSetup = None,
        user_functions: Dict[str, Callable] = None,
    ):
        self._run_parameters = run_parameters or ControllerRunParameters()
        self._devices = DeviceCollection(
            device_setup,
            self._run_parameters.dry_run,
            self._run_parameters.ignore_lab_one_version_error,
        )
        self._connected = False
        # Waves which are uploaded to the devices via pulse replacements
        self._current_waves = []
        self._user_functions: Dict[str, Callable] = user_functions
        self._nodes_from_user_functions: List[DaqNodeAction] = []
        self._recipe_data: RecipeData = None
        self._session = None
        self._results: Results = None

        self._logger = logging.getLogger(__name__)
        self._logger.debug("Controller created")
        self._logger.debug("Controller debug logging is on")

        self._logger.info("VERSION: laboneq %s", __version__)

        # TODO: Remove this option and support of AWG module.
        self._is_using_standalone_compiler = os.environ.get(
            "LABONEQ_STANDALONE_AWG", "1"
        ).lower() in ("1", "true")

    def _allocate_resources(self):
        self._devices.free_allocations()
        osc_params = self._recipe_data.recipe.experiment.oscillator_params
        for osc_param in sorted(osc_params, key=lambda p: p.id):
            self._devices.find_by_uid(osc_param.device_id).allocate_osc(osc_param)

    def _reset_to_idle_state(self):
        reset_nodes = []
        for _, device in self._devices.all:
            reset_nodes.extend(device.collect_reset_nodes())
        batch_set(reset_nodes)

    def _wait_for_conditions_to_start(self):
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            device.wait_for_conditions_to_start()

    def _initialize_device_outputs(self):
        nodes_to_initialize: List[DaqNodeAction] = []
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_output_initialization_nodes(
                    self._recipe_data.device_settings[initialization.device_uid],
                    initialization,
                )
            )
            nodes_to_initialize.extend(device.collect_osc_initialization_nodes())

        batch_set(nodes_to_initialize)

    def _set_nodes_before_awg_program_upload(self):
        nodes_to_initialize = []
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_awg_before_upload_nodes(
                    initialization, self._recipe_data
                )
            )
        batch_set(nodes_to_initialize)

    @tracing.trace("awg-program-handler")
    def _upload_awg_programs_standalone(self):
        @dataclass
        class UploadItem:
            awg_index: int
            seqc_code: str
            seqc_filename: str
            waves: List[Any]
            command_table: Dict[Any]
            elf: Optional[bytes]

        # Mise en place:
        awg_data: Dict[DeviceZI, List[UploadItem]] = defaultdict(list)
        recipe_data = self._recipe_data
        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )
        for initialization in recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)

            if initialization.awgs is None:
                continue

            for awg_obj in initialization.awgs:
                awg_index = awg_obj.awg
                seqc_code, waves, command_table = device.prepare_seqc(
                    awg_obj.seqc, recipe_data.compiled
                )
                awg_data[device].append(
                    UploadItem(
                        awg_index, seqc_code, awg_obj.seqc, waves, command_table, None
                    )
                )

        # Compile in parallel:
        def worker(device: DeviceZI, item: UploadItem, span: tracing.Span):
            with tracing.get_tracer().start_span("compile-awg-thread", span) as _:
                item.elf = device.compile_seqc(
                    item.seqc_code, item.awg_index, item.seqc_filename
                )

        self._logger.debug("Started compilation of AWG programs...")
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
        self._logger.debug("Finished compilation.")

        # Upload AWG programs, waveforms, and command tables:
        elf_node_settings: Dict[DaqWrapper, List[DaqNodeSetAction]] = defaultdict(list)
        elf_upload_conditions: Dict[DaqWrapper, Dict[str, Any]] = defaultdict(dict)
        wf_node_settings: Dict[DaqWrapper, List[DaqNodeSetAction]] = defaultdict(list)
        for device, items in awg_data.items():
            for item in items:
                elf_filename = item.seqc_filename.rsplit(".seqc", 1)[0] + ".elf"
                set_action = device.prepare_upload_elf(
                    item.elf, item.awg_index, elf_filename
                )
                node_settings = elf_node_settings[device.daq]
                node_settings.append(set_action)

                if isinstance(device, DeviceUHFQA):
                    # UHFQA does not yet support upload of ELF and waveforms in
                    # a single transaction.
                    ready_node = device.get_sequencer_paths(item.awg_index)["ready"]
                    elf_upload_conditions[device.daq][ready_node] = 1
                    node_settings = wf_node_settings[device.daq]

                node_settings += device.prepare_upload_all_binary_waves(
                    item.awg_index, item.waves, acquisition_type
                )

                if item.command_table is not None:
                    set_action = device.prepare_upload_command_table(
                        item.awg_index, item.command_table
                    )
                    node_settings.append(set_action)

        if len(elf_upload_conditions) > 0:
            for daq in elf_upload_conditions.keys():
                daq.node_monitor.flush()

        self._logger.debug("Started upload of AWG programs...")
        with tracing.get_tracer().start_span("upload-awg-programs") as _:
            for daq, nodes in elf_node_settings.items():
                daq.batch_set(nodes)

            if len(elf_upload_conditions) > 0:
                self._logger.debug("Waiting for devices...")
                response_waiter = ResponseWaiter()
                for daq, conditions in elf_upload_conditions.items():
                    response_waiter.add(
                        target=daq.node_monitor,
                        conditions=conditions,
                    )
                timeout_s = 10
                if not response_waiter.wait_all(timeout=timeout_s):
                    raise LabOneQControllerException(
                        f"AWGs not in ready state within timeout ({timeout_s} s)."
                    )

                self._logger.debug("Started upload of waveforms...")
                with tracing.get_tracer().start_span("upload-waveforms") as _:
                    for daq, nodes in wf_node_settings.items():
                        daq.batch_set(nodes)
        self._logger.debug("Finished upload.")

    def _upload_awg_programs(self):
        if self._is_using_standalone_compiler:
            return self._upload_awg_programs_standalone()

        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            device.upload_awg_program(initialization, self._recipe_data)

    def _set_nodes_after_awg_program_upload(self):
        nodes_to_initialize = []
        for initialization in self._recipe_data.initializations:
            device = self._devices.find_by_uid(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_awg_after_upload_nodes(initialization)
            )

        batch_set(nodes_to_initialize)

    def _initialize_awgs(self):
        self._set_nodes_before_awg_program_upload()
        self._upload_awg_programs()
        self._set_nodes_after_awg_program_upload()

    def _configure_leaders(self):
        self._logger.debug(
            "Using %s as leaders.",
            [d.dev_repr for _, d in self._devices.leaders],
        )
        for uid, device in self._devices.leaders:
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            device.configure_as_leader(init)

    def _configure_followers(self):
        self._logger.debug(
            "Using %s as followers.",
            [d.dev_repr for _, d in self._devices.followers],
        )
        nodes_to_configure_followers = []

        for uid, device in self._devices.followers:
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            nodes_to_configure_followers.extend(
                device.collect_follower_configuration_nodes(init)
            )

        batch_set(nodes_to_configure_followers)

    def _configure_triggers(self):
        nodes_to_configure_triggers = []

        for uid, device in itertools.chain(
            self._devices.leaders, self._devices.followers
        ):
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            nodes_to_configure_triggers.extend(
                device.collect_trigger_configuration_nodes(init, self._recipe_data)
            )

        batch_set(nodes_to_configure_triggers)

    def _initialize_devices(self):
        self._reset_to_idle_state()
        self._allocate_resources()
        self._initialize_device_outputs()
        self._initialize_awgs()
        self._configure_leaders()
        self._configure_followers()
        self._configure_triggers()
        self._wait_for_conditions_to_start()

    def _execute_one_step_followers(self):
        self._logger.debug("Settings nodes to start on followers")

        nodes_to_execute = []
        for _, device in self._devices.followers:
            nodes_to_execute.extend(device.collect_execution_nodes())

        batch_set(nodes_to_execute)

        response_waiter = ResponseWaiter()
        for _, device in self._devices.followers:
            response_waiter.add(
                target=device.daq.node_monitor,
                conditions=device.conditions_for_execution_ready(),
            )
        if not response_waiter.wait_all(timeout=2):
            self._logger.warning(
                "Conditions to start RT on followers still not fulfilled after 2 seconds, "
                "nonetheless trying to continue..."
            )

        # Standalone workaround: The device is triggering itself,
        # thus split the execution into AWG trigger arming and triggering
        nodes_to_execute = []
        for _, device in self._devices.followers:
            nodes_to_execute.extend(device.collect_start_execution_nodes())

        batch_set(nodes_to_execute)

    def _execute_one_step_leaders(self):
        self._logger.debug("Settings nodes to start on leaders")
        nodes_to_execute = []

        for _, device in self._devices.leaders:
            nodes_to_execute.extend(device.collect_execution_nodes())
        batch_set(nodes_to_execute)

    def _wait_execution_to_stop(self, acquisition_type: AcquisitionType):
        min_wait_time = self._recipe_data.recipe.experiment.total_execution_time
        if min_wait_time is None:
            self._logger.warning(
                "No estimation available for the execution time, assuming 10 sec."
            )
            min_wait_time = 10.0
        elif min_wait_time > 5:  # Only inform about RT executions taking longer than 5s
            self._logger.info("Estimated RT execution time: %.2f s.", min_wait_time)
        guarded_wait_time = round(
            min_wait_time * 1.1 + 1
        )  # +10% and fixed 1sec guard time

        response_waiter = ResponseWaiter()
        for _, device in self._devices.followers:
            response_waiter.add(
                target=device.daq.node_monitor,
                conditions=device.conditions_for_execution_done(acquisition_type),
            )
        if not response_waiter.wait_all(timeout=guarded_wait_time):
            self._logger.warning(
                "Stop conditions still not fulfilled after %f s, estimated execution time "
                "was %.2f s. Continuing to the next step.",
                guarded_wait_time,
                min_wait_time,
            )

    def _execute_one_step(self, acquisition_type: AcquisitionType):
        self._logger.debug("Step executing")

        self._devices.flush_monitor()

        # Can't batch everything together, because PQSC needs to be executed after HDs
        # otherwise it can finish before AWGs are started, and the trigger is lost
        self._execute_one_step_followers()
        self._execute_one_step_leaders()

        self._logger.debug("Execution started")

        self._wait_execution_to_stop(acquisition_type)

        self._logger.debug("Execution stopped")

    def connect(self):
        if not self._connected:
            self._devices.connect(self._is_using_standalone_compiler)
            self._connected = True

    def shut_down(self):
        self._logger.info("Shutting down all devices...")
        self._devices.shut_down()
        self._logger.info("Successfully Shut down all devices.")

    def disconnect(self):
        self._logger.info("Disconnecting from all devices and servers...")
        self._devices.disconnect()
        self._logger.info("Successfully disconnected from all devices and servers.")
        self._connected = False

    def execute_compiled(
        self, compiled_experiment: CompiledExperiment, session: Session = None
    ):
        self._recipe_data = pre_process_compiled(compiled_experiment)

        self._session = session
        if session is None:
            self._results = None
        else:
            self._results = session._last_results

        self.connect()
        self._prepare_result_shapes()
        try:
            self._devices.start_monitor()
            self._initialize_devices()

            # Ensure no side effects from the previous execution in the same session
            self._current_waves = []
            self._nodes_from_user_functions = []
            self._logger.info("Starting near-time execution...")
            with tracing.get_tracer().start_span("near-time-execution"):
                Controller.NearTimeExecutor(controller=self).run(
                    self._recipe_data.execution
                )
            self._logger.info("Finished near-time execution.")
            for _, device in self._devices.all:
                device.check_errors()
        finally:
            self._devices.stop_monitor()

        if self._run_parameters.shut_down is True:
            self.shut_down()

        if self._run_parameters.disconnect is True:
            self.disconnect()

    def _find_awg(self, seqc_name: str) -> Tuple[str, int]:
        # TODO(2K): Do this in the recipe preprocessor, or even modify the compiled experiment
        # data model
        for init in self._recipe_data.initializations:
            if init.awgs is None:
                continue
            for awg in init.awgs:
                if awg.seqc == seqc_name:
                    return init.device_uid, awg.awg
        return None, None

    def replace_pulse(
        self, pulse_uid: Union[str, Pulse], pulse_or_array: Union[npt.ArrayLike, Pulse]
    ):
        """Replaces specific pulse with the new sample data on the device.

        This is useful when called from the user function, allows fast waveform replacement within
        near-time loop without experiment recompilation.

        Args:
            pulse_uid: pulse to replace, can be Pulse object or uid of the pulse
            pulse_or_array: replacement pulse, can be Pulse object or value array
            (see sampled_pulse_* from the pulse library)
        """
        acquisition_type = RtExecutionInfo.get_acquisition_type(
            self._recipe_data.rt_execution_infos
        )
        wave_replacements = calc_wave_replacements(
            self._recipe_data.compiled, pulse_uid, pulse_or_array, self._current_waves
        )
        for repl in wave_replacements:
            awg_indices = next(
                a
                for a in self._recipe_data.compiled.wave_indices
                if a["filename"] == repl.awg_id
            )
            awg_wave_map: Dict[str, List[Union[int, str]]] = awg_indices["value"]
            target_wave = awg_wave_map.get(repl.sig_string)
            seqc_name = repl.awg_id[: -len("_waveindices.csv")] + ".seqc"
            awg = self._find_awg(seqc_name)
            device = self._devices.find_by_uid(awg[0])

            if repl.replacement_type == ReplacementType.I_Q:
                clipped = np.clip(repl.samples, -1.0, 1.0)
                bin_wave = zhinst.utils.convert_awg_waveform(*clipped)
                self._nodes_from_user_functions.append(
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
                self._nodes_from_user_functions.append(
                    device.prepare_upload_binary_wave(
                        filename=repl.sig_string + " (repl)",
                        waveform=repl.samples,
                        awg_index=awg[1],
                        wave_index=target_wave[0],
                        acquisition_type=acquisition_type,
                    )
                )

    def _prepare_rt_execution(
        self, rt_section_uid: str
    ) -> Tuple[List[DaqNodeAction], AcquisitionType]:
        if rt_section_uid is None:
            return [], []  # Old recipe-based execution - skip RT preparation
        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]
        self._nodes_from_user_functions.sort(key=lambda v: v.path)
        nodes_to_prepare_rt = [*self._nodes_from_user_functions]
        self._nodes_from_user_functions.clear()
        for awg_key, awg_config in rt_execution_info.per_awg_configs.items():
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
                    self._recipe_data.recipe.experiment.integrator_allocations,
                    effective_averages,
                    effective_averaging_mode,
                    rt_execution_info.acquisition_type,
                )
            )
        return nodes_to_prepare_rt

    class NearTimeExecutor(ExecutorBase):
        def __init__(self, controller: Controller):
            super().__init__(looping_mode=LoopingMode.EXECUTE)
            self.controller = controller
            self.step_param_nodes = []
            self.nt_loop_indices: List[int] = []

        def set_handler(self, path: str, value):
            dev = self.controller._devices.find_by_path(path)
            self.step_param_nodes.append(
                DaqNodeSetAction(
                    dev._daq, path, value, caching_strategy=CachingStrategy.NO_CACHE
                )
            )

        def user_func_handler(self, func_name: str, args: Dict[str, Any]):
            func = self.controller._user_functions.get(func_name)
            if func is None:
                raise LabOneQControllerException(
                    f"User function '{func_name}' is not registered."
                )
            res = func(ProtectedSession(self.controller._session), **args)
            user_func_results = self.controller._results.user_func_results.setdefault(
                func_name, []
            )
            user_func_results.append(res)

        def set_sw_param_handler(
            self,
            name: str,
            index: int,
            value: float,
            axis_name: str,
            values: npt.ArrayLike,
        ):
            device_ids = self.controller._recipe_data.param_to_device_map.get(name, [])
            for device_id in device_ids:
                device = self.controller._devices.find_by_uid(device_id)
                self.step_param_nodes.extend(
                    device.collect_prepare_sweep_step_nodes_for_param(name, value)
                )

        def for_loop_handler(self, count: int, index: int, enter: bool):
            if enter:
                self.nt_loop_indices.append(index)
            else:
                self.nt_loop_indices.pop()

        def rt_handler(
            self,
            count: int,
            uid: str,
            averaging_mode: AveragingMode,
            acquisition_type: AcquisitionType,
            enter: bool,
        ):
            if enter:
                step_prepare_nodes = self.controller._prepare_rt_execution(
                    rt_section_uid=uid
                )
                batch_set([*self.step_param_nodes, *step_prepare_nodes])
                self.step_param_nodes.clear()
                for retry in range(3):  # Up to 3 retries
                    if retry > 0:
                        self.controller._logger.info("Step retry %s of 3...", retry + 1)
                        batch_set(step_prepare_nodes)
                    try:
                        self.controller._execute_one_step(acquisition_type)
                        self.controller._read_one_step_results(
                            nt_loop_indices=self.nt_loop_indices, rt_section_uid=uid
                        )
                        break
                    except LabOneQControllerException:  # TODO(2K): introduce "hard" controller exceptions
                        self.controller._report_step_error(
                            nt_loop_indices=self.nt_loop_indices,
                            rt_section_uid=uid,
                            message=traceback.format_exc(),
                        )

    def _prepare_result_shapes(self):
        if self._results is None:
            self._results = make_empty_results()
        if len(self._recipe_data.rt_execution_infos) == 0:
            return
        if len(self._recipe_data.rt_execution_infos) > 1:
            raise LabOneQControllerException(
                "Multiple 'acquire_loop_rt' sections per experiment is not supported."
            )
        rt_info = next(r for r in self._recipe_data.rt_execution_infos.values())
        awg_config = next((c for c in rt_info.per_awg_configs.values()), None)
        # Use default length 4096, in case AWG config is not available
        acquire_length = 4096 if awg_config is None else awg_config.acquire_length
        for handle, shape_info in self._recipe_data.result_shapes.items():
            if rt_info.acquisition_type == AcquisitionType.RAW:
                if len(self._recipe_data.result_shapes) > 1:
                    raise LabOneQControllerException(
                        f"Multiple raw acquire events with handles "
                        f"{list(self._recipe_data.result_shapes.keys())}. "
                        f"Only single raw acquire per experiment allowed."
                    )
                empty_res = make_acquired_result(
                    data=np.empty(shape=[acquire_length], dtype=np.complex128),
                    axis_name=["samples"],
                    axis=[np.arange(acquire_length)],
                )
                empty_res.data[:] = np.nan
                self._results.acquired_results[handle] = empty_res
                return  # Only one result supported in RAW mode
            axis_name = deepcopy(shape_info.base_axis_name)
            axis = deepcopy(shape_info.base_axis)
            shape = deepcopy(shape_info.base_shape)
            if shape_info.additional_axis > 1:
                axis_name.append(handle)
                axis.append(
                    np.linspace(
                        0, shape_info.additional_axis - 1, shape_info.additional_axis
                    )
                )
                shape.append(shape_info.additional_axis)
            empty_res = make_acquired_result(
                data=np.empty(shape=tuple(shape), dtype=np.complex128),
                axis_name=axis_name,
                axis=axis,
            )
            if len(shape) == 0:
                empty_res.data = np.nan
            else:
                empty_res.data[:] = np.nan
            self._results.acquired_results[handle] = empty_res

    def _read_one_step_results(self, nt_loop_indices: List[int], rt_section_uid: str):
        if rt_section_uid is None:
            return  # Old recipe-based execution - skip partial result processing
        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]
        for awg_key, awg_config in rt_execution_info.per_awg_configs.items():
            device = self._devices.find_by_uid(awg_key.device_uid)
            if rt_execution_info.acquisition_type == AcquisitionType.RAW:
                raw_results = device.get_input_monitor_data(
                    awg_key.awg_index, awg_config.acquire_length
                )
                # Copy to all result handles, but actually only one handle is supported for now
                for signal in awg_config.signals:
                    mapping = rt_execution_info.signal_result_map.get(signal, [])
                    unique_handles = set(mapping)
                    for handle in unique_handles:
                        result = self._results.acquired_results[handle]
                        for raw_result_idx, raw_result in enumerate(raw_results):
                            result.data[raw_result_idx] = raw_result
            else:
                if rt_execution_info.averaging_mode == AveragingMode.SINGLE_SHOT:
                    effective_averages = 1
                else:
                    effective_averages = rt_execution_info.averages
                device.check_results_acquired_status(
                    awg_key.awg_index,
                    rt_execution_info.acquisition_type,
                    awg_config.result_length,
                    effective_averages,
                )
                for signal in awg_config.signals:
                    integrator_allocation = next(
                        i
                        for i in self._recipe_data.recipe.experiment.integrator_allocations
                        if i.signal_id == signal
                    )
                    assert integrator_allocation.device_id == awg_key.device_uid
                    assert integrator_allocation.awg == awg_key.awg_index
                    result_indices = integrator_allocation.channels
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
                            result, nt_loop_indices, raw_results, mapping, handle
                        )

    def _report_step_error(
        self, nt_loop_indices: List[int], rt_section_uid: str, message: str
    ):
        self._results.execution_errors.append(
            (deepcopy(nt_loop_indices), rt_section_uid, message)
        )
