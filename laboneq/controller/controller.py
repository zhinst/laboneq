# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from copy import deepcopy
import itertools
import logging
import re
import traceback
import numpy as np

from typing import Generator, List, Dict, Any, Callable, Tuple, Union, TYPE_CHECKING
from numpy import typing as npt

import zhinst.utils

from pkg_resources import get_distribution
from laboneq.controller.protected_session import ProtectedSession

from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements

from laboneq.executor.executor import ExecutorBase, LoopingMode

from .devices.device_base import DeviceBase, DeviceQualifier
from .devices.device_factory import DeviceFactory
from .recipe_1_4_0 import *
from .recipe_processor import RecipeData, RtExecutionInfo, pre_process_compiled
from .util import LabOneQControllerException
from .results import make_empty_results, make_acquired_result, build_partial_result
from .communication import (
    DaqNodeGetAction,
    DaqNodeAction,
    DaqNodeSetAction,
    CachingStrategy,
    DaqWrapper,
    DaqWrapperDryRun,
    ServerQualifier,
)

from .cache import CacheTreeNode

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl import Session
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.instrument import Instrument
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
        self._device_setup: DeviceSetup = device_setup
        self._connected = False
        self._user_functions: Dict[str, Callable] = user_functions
        self._nodes_from_user_functions: List[DaqNodeAction] = []
        self._recipe_data: RecipeData = None
        self._session = None
        self._results: Results = None
        self.cache_to_inject = None
        self._daqs: Dict[str, DaqWrapper] = {}
        self._devices: Dict[str, DeviceBase] = {}
        self._logger = logging.getLogger(__name__)
        self._logger.debug("Controller created")
        self._logger.debug("Controller debug logging is on")
        self._current_waves = []
        "Waves which are uploaded to the devices via pulse replacements"

        version = get_distribution("laboneq").version
        self._logger.info("VERSION: laboneq %s", version)

    DevIterType = Generator[Tuple[str, DeviceBase], None, None]

    @property
    def leaders(self) -> DevIterType:
        for uid, device in self._devices.items():
            if device.is_leader():
                yield uid, device

    @property
    def followers(self) -> DevIterType:
        for uid, device in self._devices.items():
            if device.is_follower():
                yield uid, device

    @property
    def devices(self) -> Dict[str, DeviceBase]:
        """Controller devices"""
        return self._devices

    def _prepare_devices(self, instruments: List[Instrument]):
        def make_device_qualifier(dry_run: bool, instrument, daq) -> DeviceQualifier:

            driver = instrument.calc_driver()
            options = instrument.calc_options()

            return DeviceQualifier(
                dry_run=dry_run, driver=driver, server=daq, options=options
            )

        updated_devices: Dict[str, DeviceBase] = {}
        for instrument in instruments:
            if hasattr(instrument, "server_uid"):
                daq = self._daqs.get(instrument.server_uid)
            else:
                daq = None
            device_qualifier = make_device_qualifier(
                self._run_parameters.dry_run, instrument, daq
            )
            device = self._devices.get(instrument.uid)
            if device is None or device.device_qualifier != device_qualifier:
                device = DeviceFactory.create(device_qualifier)
            device.remove_all_links()
            updated_devices[instrument.uid] = device
        self._devices = updated_devices

        # Update device links
        for instrument in instruments:
            from_dev = self._devices[instrument.uid]
            for connection in instrument.connections:
                if connection.signal_type in [IOSignalType.DIO, IOSignalType.ZSYNC]:
                    from_port = connection.local_port
                    to_dev = self._devices.get(connection.remote_path)
                    if to_dev is None:
                        raise LabOneQControllerException(
                            f"Could not find destination device '{connection.remote_path}' for the port '{connection.local_port}' connection of the device '{instrument.uid}'"
                        )
                    to_port = f"{connection.signal_type.name}/{connection.remote_port}"
                    from_dev.add_downlink(from_port, to_dev)
                    to_dev.add_uplink(to_port, from_dev)

    def _connect_devices(self):
        for device in self._devices.values():
            device.connect()

    def _find_device(self, device_uid):
        device = self._devices.get(device_uid)
        if device is None:
            raise LabOneQControllerException(
                f"Could not find device object for the device uid '{device_uid}'"
            )
        return device

    def _find_device_by_path(self, path: str):
        m = re.match(r"^/?(DEV\d+)/.+", path.upper())
        if m is None:
            raise LabOneQControllerException(
                f"Path '{path}' is not referring to any device"
            )
        serial = m.group(1)
        dev: DeviceBase
        for dev in self._devices.values():
            if dev._get_option("serial").upper() == serial:
                return dev
        raise LabOneQControllerException(f"Could not find device for the path '{path}'")

    def _sync_all_daqs(self):
        for daq in self._daqs.values():
            daq.sync()

    def _allocate_resources(self):
        for device in self._devices.values():
            device.free_allocations()
        for osc_param in self._recipe_data.recipe.experiment.oscillator_params:
            self._devices[osc_param.device_id].allocate_osc(osc_param)

    def _reset_to_idle_state(self):
        reset_nodes = []
        for device in self._devices.values():
            reset_nodes.extend(device.collect_reset_nodes())
        self._batch_set(reset_nodes)
        self._sync_all_daqs()

    def _wait_for_conditions_to_start(self):
        for initialization in self._recipe_data.initializations:
            device = self._find_device(initialization.device_uid)
            device.wait_for_conditions_to_start()

    def _initialize_device_outputs(self):
        nodes_to_initialize: List[DaqNodeAction] = []
        for initialization in self._recipe_data.initializations:
            device = self._find_device(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_output_initialization_nodes(
                    self._recipe_data.device_settings[initialization.device_uid],
                    initialization,
                )
            )
            nodes_to_initialize.extend(device.collect_osc_initialization_nodes())

        self._batch_set(nodes_to_initialize)

    def _set_nodes_before_awg_program_upload(self):
        nodes_to_initialize = []
        for initialization in self._recipe_data.initializations:
            device = self._find_device(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_awg_before_upload_nodes(
                    initialization, self._recipe_data
                )
            )
        self._batch_set(nodes_to_initialize)

    def _upload_awg_programs(self):
        for initialization in self._recipe_data.initializations:
            device = self._find_device(initialization.device_uid)
            device.upload_awg_program(initialization, self._recipe_data)

    def _set_nodes_after_awg_program_upload(self):
        nodes_to_initialize = []
        for initialization in self._recipe_data.initializations:
            device = self._find_device(initialization.device_uid)
            nodes_to_initialize.extend(
                device.collect_awg_after_upload_nodes(initialization)
            )

        self._batch_set(nodes_to_initialize)

    def _initialize_awgs(self):
        self._set_nodes_before_awg_program_upload()
        self._upload_awg_programs()
        self._set_nodes_after_awg_program_upload()

    def _configure_leaders(self):
        self._logger.debug(
            "Using %s as leaders.", [d.dev_repr for _, d in self.leaders]
        )
        for uid, device in self.leaders:
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            device.configure_as_leader(init)

    def _configure_followers(self):
        self._logger.debug(
            "Using %s as followers.", [d.dev_repr for _, d in self.followers]
        )
        nodes_to_configure_followers = []

        for uid, device in self.followers:
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            nodes_to_configure_followers.extend(
                device.collect_follower_configuration_nodes(init)
            )

        self._batch_set(nodes_to_configure_followers)

        self._sync_all_daqs()

    def _configure_triggers(self):
        nodes_to_configure_triggers = []

        for uid, device in itertools.chain(self.leaders, self.followers):
            init = self._recipe_data.get_initialization_by_device_uid(uid)
            if init is None:
                continue
            nodes_to_configure_triggers.extend(
                device.collect_trigger_configuration_nodes(init)
            )

        self._batch_set(nodes_to_configure_triggers)

    def _initialize_devices(self):
        self._reset_to_idle_state()
        self._allocate_resources()
        self._initialize_device_outputs()
        self._initialize_awgs()
        self._configure_leaders()
        self._configure_followers()
        self._configure_triggers()
        self._sync_all_daqs()
        self._wait_for_conditions_to_start()

    def _split_daq_actions(
        self, daq_actions: List[DaqNodeAction]
    ) -> Dict[DaqWrapper, List[DaqNodeAction]]:
        res: Dict[DaqWrapper, List[DaqNodeAction]] = {}
        for daq_action in daq_actions:
            if not daq_action.daq in res:
                res[daq_action.daq] = []

            res[daq_action.daq].append(daq_action)
        return res

    def _wait_all_conditions(self, wait_conditions):
        split_actions: Dict[DaqWrapper, List[DaqNodeAction]] = self._split_daq_actions(
            wait_conditions
        )
        for daq, actions in split_actions.items():
            daq.wait_all_conditions(
                actions, self._recipe_data.recipe.experiment.total_execution_time
            )

    def _batch_set(self, daq_actions: List[DaqNodeAction]):
        split_actions: Dict[DaqWrapper, List[DaqNodeAction]] = self._split_daq_actions(
            daq_actions
        )
        for daq in split_actions:
            daq.batch_set(split_actions[daq])

    def set(self, path: str, value: Any):
        self._check_connected()
        dev = self._find_device_by_path(path)
        daq: DaqWrapper = dev._daq
        daq.batch_set(
            [
                DaqNodeSetAction(
                    daq, path, value, caching_strategy=CachingStrategy.NO_CACHE
                )
            ]
        )

    def get(self, path: str) -> Any:
        self._check_connected()
        dev = self._find_device_by_path(path)
        daq: DaqWrapper = dev._daq
        return daq.get(DaqNodeGetAction(daq, path))

    def _execute_one_step_followers(self):
        self._logger.debug("Settings nodes to start on followers")

        nodes_to_execute = []
        for _, device in self.followers:
            nodes_to_execute.extend(device.collect_execution_nodes())

        self._batch_set(nodes_to_execute)

        for _, device in self.followers:
            device.wait_for_execution_ready()

    def _execute_one_step_leaders(self):
        self._logger.debug("Settings nodes to start on leaders")
        nodes_to_execute = []

        for _, device in self.leaders:
            nodes_to_execute.extend(device.collect_execution_nodes())
        self._batch_set(nodes_to_execute)

    def _wait_execution_to_stop(
        self, acquisition_units: List[Tuple[str, int, AcquisitionType]]
    ):
        self._logger.debug("Waiting for execution stop")

        wait_conditions_to_end = []
        for uid, device in itertools.chain(self.followers, self.leaders):
            wait_conditions_to_end.extend(
                device.collect_conditions_to_close_loop(
                    [
                        (awg, acq_type)
                        for (dev_uid, awg, acq_type) in acquisition_units
                        if dev_uid == uid
                    ]
                )
            )

        self._wait_all_conditions(wait_conditions_to_end)

    def _prepare_end_conditions(
        self, acquisition_units: List[Tuple[str, int, AcquisitionType]]
    ):
        self._logger.debug("Preparing end conditions")

        wait_conditions_to_end = []
        for uid, device in itertools.chain(self.followers, self.leaders):
            wait_conditions_to_end.extend(
                device.collect_conditions_to_close_loop(
                    [
                        (awg, acq_type)
                        for (dev_uid, awg, acq_type) in acquisition_units
                        if dev_uid == uid
                    ]
                )
            )

        self._prepare_wait_conditions(wait_conditions_to_end)

    def _prepare_wait_conditions(self, wait_conditions):
        split_actions: Dict[DaqWrapper, List[DaqNodeAction]] = self._split_daq_actions(
            wait_conditions
        )
        for daq in split_actions:
            daq.prepare_conditions(split_actions[daq])

    def _execute_one_step(
        self, acquisition_units: List[Tuple[str, int, AcquisitionType]]
    ):
        self._logger.debug("Step executing")

        # Can't batch everything together, because PQSC needs to be executed after HDs
        # otherwise it can finish before AWGs are started, and the trigger is lost
        self._execute_one_step_followers()
        self._execute_one_step_leaders()

        self._logger.debug("Execution started")

        self._wait_execution_to_stop(acquisition_units)

        self._logger.debug("Execution stopped")

    def connect(self):
        self._create_daqs(self._device_setup)
        self._prepare_devices(self._device_setup.instruments)
        self._connect_devices()
        self._connected = True

    def _check_connected(self):
        if not self._connected:
            self.connect()

    def shut_down(self):
        self._logger.info("Shutting down all devices...")
        for device in self._devices.values():
            device.shut_down()
        self._logger.info("Successfully Shut down all devices.")

    def disconnect(self):
        self._logger.info("Disconnecting from all devices...")
        for device in self._devices.values():
            device.disconnect()
        self._devices = {}
        self._logger.info("Successfully disconnected from all devices.")
        self._daqs = {}
        self._logger.info("Disconnected from all servers.")
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

        self._check_connected()
        self._prepare_result_shapes()
        self._initialize_devices()
        self._current_waves = []
        self._logger.info("Starting near-time execution...")
        # Ensure no side effects from the previous execution in the same session
        self._nodes_from_user_functions = []
        Controller.NearTimeExecutor(controller=self).run(self._recipe_data.execution)
        for device in self._devices.values():
            device.check_errors()
        self._logger.info("Finished near-time execution.")

        if self._run_parameters.shut_down is True:
            self.shut_down()

        if self._run_parameters.disconnect is True:
            self.disconnect()

    def _find_awg(self, seqc_name: str) -> Tuple[str, int]:
        # TODO(2K): Do this in the recipe preprocessor, or even modify the compiled experiment data model
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

        This is useful when called from the user function, allows fast waveform replacement within near-time
        loop without experiment recompilation.

        Args:
            pulse_uid: pulse to replace, can be Pulse object or uid of the pulse
            pulse_or_array: replacement pulse, can be Pulse object or value array (see sampled_pulse_* from the pulse library)
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
            awg_wave_map = awg_indices["value"]
            target_wave = awg_wave_map.get(repl.sig_string)
            seqc_name = repl.awg_id[: -len("_waveindices.csv")] + ".seqc"
            awg = self._find_awg(seqc_name)
            device = self._find_device(awg[0])

            if repl.replacement_type == ReplacementType.I_Q:
                bin_wave = zhinst.utils.convert_awg_waveform(
                    repl.samples[0], repl.samples[1]
                )
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
                self._nodes_from_user_functions.append(
                    device.prepare_upload_binary_wave(
                        filename=repl.sig_string + " (repl)",
                        waveform=repl.samples,
                        awg_index=awg[1],
                        wave_index=target_wave[0],
                        acquisition_type=acquisition_type,
                    )
                )

    def _create_daqs(self, device_setup: DeviceSetup):
        servers = device_setup.servers

        self._logger.debug("Creating/updating data servers connections")

        def make_server_qualifier(
            dry_run: bool, server=None, ignore_lab_one_version_error=False
        ):
            from laboneq.dsl.device.servers import DataServer

            if isinstance(server, DataServer):
                return ServerQualifier(
                    dry_run=dry_run,
                    host=server.host,
                    port=int(server.port),
                    api_level=int(server.api_level),
                    ignore_lab_one_version_error=ignore_lab_one_version_error,
                )
            else:
                self._logger.warning(
                    "Server provider '%s' is not supported by the controller.",
                    type(server).__name__,
                )

        updated_daqs: Dict[str, DaqWrapper] = {}
        for server_uid, server in servers.items():
            server_qualifier = make_server_qualifier(
                self._run_parameters.dry_run,
                server=server,
                ignore_lab_one_version_error=self._run_parameters.ignore_lab_one_version_error,
            )
            existing = self._daqs.get(server_uid)
            if existing is not None and existing.server_qualifier == server_qualifier:
                updated_daqs[server_uid] = existing
                continue

            self._logger.info(
                "Connecting to data server at %s:%s", server.host, server.port
            )
            if server_qualifier.dry_run:
                daq = DaqWrapperDryRun(server_uid, server_qualifier)
                for instr in device_setup.instruments:
                    daq.map_device_type(instr.address, instr.calc_driver())
            else:
                daq = DaqWrapper(server_uid, server_qualifier)
            updated_daqs[server_uid] = daq
        self._daqs = updated_daqs

    def _inject_cache_internal(self):
        if self.cache_to_inject is None:
            return
        for server_uid in self.cache_to_inject.children:
            if not server_uid in self._daqs:
                self._logger.error(self.cache_to_inject.children.keys())
                continue
            self._daqs[server_uid].inject_cache_tree(
                self.cache_to_inject.children[server_uid]
            )

    def inject_cache_tree(self, cache_tree_node):
        self.cache_to_inject = cache_tree_node
        self._inject_cache_internal()

    def extract_cache_tree(self):
        cache_tree_root = CacheTreeNode(None)
        for server_uid in self._daqs:
            cache_tree_root.add_child(
                server_uid, self._daqs[server_uid].extract_cache_tree()
            )
        return cache_tree_root

    def _prepare_rt_execution(
        self, rt_section_uid: str
    ) -> Tuple[List[DaqNodeAction], List[Tuple[str, int, AcquisitionType]]]:
        if rt_section_uid is None:
            return [], []  # Old recipe-based execution - skip RT preparation
        rt_execution_info = self._recipe_data.rt_execution_infos[rt_section_uid]
        self._nodes_from_user_functions.sort(key=lambda v: v.path)
        nodes_to_prepare_rt = [*self._nodes_from_user_functions]
        self._nodes_from_user_functions.clear()
        acquisition_units = []
        for awg_key, awg_config in rt_execution_info.per_awg_configs.items():
            device = self._devices[awg_key.device_uid]
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
            acquisition_units.append(
                (
                    awg_key.device_uid,
                    awg_key.awg_index,
                    rt_execution_info.acquisition_type,
                )
            )
        return nodes_to_prepare_rt, acquisition_units

    class NearTimeExecutor(ExecutorBase):
        def __init__(self, controller: Controller):
            super().__init__(looping_mode=LoopingMode.EXECUTE)
            self.controller = controller
            self.nodes_to_prepare_step = []
            self.nt_loop_indices: List[int] = []

        def set_handler(self, path: str, val):
            dev = self.controller._find_device_by_path(path)
            self.nodes_to_prepare_step.append(
                DaqNodeSetAction(
                    dev._daq, path, val, caching_strategy=CachingStrategy.NO_CACHE
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
            val: float,
            axis_name: str,
            values: npt.ArrayLike,
        ):
            device_ids = self.controller._recipe_data.param_to_device_map.get(name, [])
            for device_id in device_ids:
                device = self.controller._find_device(device_id)
                self.nodes_to_prepare_step.extend(
                    device.collect_prepare_sweep_step_nodes_for_param(name, val)
                )

        def for_loop_handler(self, count: int, index: int, enter: bool):
            if enter:
                self.nt_loop_indices.append(index)
            else:
                self.nt_loop_indices.pop()

        def rt_handler(
            self, count: int, uid: str, averaging_mode, acquisition_type, enter: bool
        ):
            if enter:
                step_nodes, acquisition_units = self.controller._prepare_rt_execution(
                    rt_section_uid=uid
                )
                self.nodes_to_prepare_step.extend(step_nodes)
                self.controller._batch_set(self.nodes_to_prepare_step)
                self.nodes_to_prepare_step.clear()
                for _ in range(3):  # Up to 3 retries
                    try:
                        self.controller._prepare_end_conditions(acquisition_units)
                        self.controller._execute_one_step(acquisition_units)
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
                        f"Multiple raw acquire events with handles {list(self._recipe_data.result_shapes.keys())}. Only single raw acquire per experiment allowed."
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
            device = self._devices[awg_key.device_uid]
            if rt_execution_info.acquisition_type == AcquisitionType.RAW:
                raw_result = device.get_input_monitor_data(
                    awg_key.awg_index, awg_config.acquire_length
                )
                # Copy to all result handles, but actually only one handle is supported for now
                for signal in awg_config.signals:
                    mapping = rt_execution_info.signal_result_map.get(signal, [])
                    unique_handles = set(mapping)
                    for handle in unique_handles:
                        result = self._results.acquired_results[handle]
                        for raw_result_idx in range(len(raw_result)):
                            result.data[raw_result_idx] = raw_result[raw_result_idx]
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
                    raw_result = device.get_measurement_data(
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
                            result, nt_loop_indices, raw_result, mapping, handle,
                        )

    def _report_step_error(
        self, nt_loop_indices: List[int], rt_section_uid: str, message: str
    ):
        self._results.execution_errors.append(
            (deepcopy(nt_loop_indices), rt_section_uid, message)
        )
