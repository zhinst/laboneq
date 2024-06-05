# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections import defaultdict

import json
import logging
import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator
from weakref import ReferenceType, ref

import numpy as np
from numpy.typing import ArrayLike
import zhinst.core
import zhinst.utils
from numpy import typing as npt

from laboneq.controller.attribute_value_tracker import (  # pylint: disable=E0401
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeSetAction,
    DaqWrapper,
)
from laboneq.controller.devices.async_support import (
    NodeMonitorAsync,
    create_device_kernel_session,
    get_raw,
    set_parallel,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.zi_emulator import EmulatorState, MockInstrument
from laboneq.controller.devices.zi_node_monitor import (
    INodeMonitorProvider,
    NodeControlBase,
    NodeMonitorBase,
    ResponseWaiter,
)
from laboneq.controller.pipeliner_reload_tracker import PipelinerReloadTracker
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    DeviceRecipeData,
    RecipeData,
    RtExecutionInfo,
    get_wave,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.utilities.seqc_compile import SeqCCompileItem, seqc_compile_async
from laboneq.core.utilities.string_sanitize import string_sanitize
from laboneq.data.recipe import (
    Initialization,
    IntegratorAllocation,
    NtStepKey,
    OscillatorParam,
)
from laboneq.data.scheduled_experiment import (
    ArtifactsCodegen,
    CodegenWaveform,
    ScheduledExperiment,
)

_logger = logging.getLogger(__name__)

seqc_osc_match = re.compile(
    r'(\s*string\s+osc_node_)(\w+)(\s*=\s*"oscs/)[0-9]+(/freq"\s*;\s*)', re.ASCII
)


class AwgCompilerStatus(Enum):
    SUCCESS = 0
    ERROR = 1
    WARNING = 2


@dataclass
class AllocatedOscillator:
    group: int
    channels: set[int]
    index: int
    id: str
    frequency: float
    param: str


@dataclass
class DeviceOptions:
    serial: str
    interface: str
    dev_type: str | None = None
    is_qc: bool | None = False
    qc_with_qa: bool = False
    gen2: bool = False
    reference_clock_source: str | None = None
    expected_installed_options: str | None = None


@dataclass
class DeviceQualifier:
    uid: str
    server_uid: str
    driver: str
    options: DeviceOptions


@dataclass
class SequencerPaths:
    elf: str
    progress: str
    enable: str
    ready: str


@dataclass
class WaveformItem:
    index: int
    name: str
    samples: npt.ArrayLike
    hold_start: int | None = None
    hold_length: int | None = None


@dataclass
class RawReadoutData:
    vector: ArrayLike

    # metadata by job_id
    metadata: dict[int, dict[str, Any]] = field(default_factory=dict)


Waveforms = list[WaveformItem]


def delay_to_rounded_samples(
    channel: int,
    dev_repr: str,
    delay: float,
    sample_frequency_hz,
    granularity_samples,
    max_node_delay_samples,
) -> int:
    if delay < 0:
        raise LabOneQControllerException(
            f"Negative node delay for device {dev_repr} and channel {channel} specified."
        )

    delay_samples = delay * sample_frequency_hz
    # Quantize to granularity and round ties towards zero
    delay_rounded = (
        math.ceil(delay_samples / granularity_samples + 0.5) - 1
    ) * granularity_samples

    if delay_rounded > max_node_delay_samples:
        raise LabOneQControllerException(
            f"Maximum delay via {dev_repr}'s node is "
            + f"{max_node_delay_samples / sample_frequency_hz * 1e9:.2f} ns - for larger "
            + "values, use the delay_signal property."
        )
    if abs(delay_samples - delay_rounded) > 1:
        _logger.debug(
            "Node delay %.2f ns of %s, channel %d will be rounded to "
            "%.2f ns, a multiple of %.0f samples.",
            delay_samples / sample_frequency_hz * 1e9,
            dev_repr,
            channel,
            delay_rounded / sample_frequency_hz * 1e9,
            granularity_samples,
        )

    return delay_rounded


class DeviceZI(INodeMonitorProvider):
    def __init__(self, device_qualifier: DeviceQualifier, daq: DaqWrapper):
        self._device_qualifier = device_qualifier
        self._downlinks: dict[str, list[tuple[str, ReferenceType[DeviceZI]]]] = {}
        self._uplinks: list[ReferenceType[DeviceZI]] = []
        self._rf_offsets: dict[int, float] = {}

        self._daq = daq
        self._api = None  # TODO(2K): Add type labone.Instrument
        self._node_monitor: NodeMonitorBase | None = None
        self.dev_type: str | None = None
        self.dev_opts: list[str] = []
        self._connected = False
        self._allocated_oscs: list[AllocatedOscillator] = []
        self._allocated_awgs: set[int] = set()
        self._pipeliner_reload_tracker: dict[int, PipelinerReloadTracker] = defaultdict(
            PipelinerReloadTracker
        )
        self._nodes_to_monitor = None
        self._sampling_rate = None
        self._device_class = 0x0

        if self._daq is None:
            raise LabOneQControllerException("ZI devices need daq")

        if self.serial is None:
            raise LabOneQControllerException(
                "ZI device must be provided with serial number via options"
            )

        if self.interface is None or self.interface == "":
            raise LabOneQControllerException(
                "ZI device must be provided with interface via options"
            )

    @property
    def device_qualifier(self):
        return self._device_qualifier

    @property
    def dev_repr(self) -> str:
        return f"{self._device_qualifier.driver.upper()}:{self.serial}"

    @property
    def has_awg(self) -> bool:
        return self._get_num_awgs() > 0

    @property
    def has_pipeliner(self) -> bool:
        return False

    @property
    def options(self) -> DeviceOptions:
        return self._device_qualifier.options

    @property
    def serial(self):
        return self.options.serial.lower()

    @property
    def interface(self):
        return self.options.interface.lower()

    @property
    def daq(self):
        return self._daq

    @property
    def node_monitor(self) -> NodeMonitorBase:
        assert self._node_monitor is not None
        return self._node_monitor

    @property
    def is_secondary(self) -> bool:
        return False

    async def maybe_async(self, nodes: NodeCollector) -> list[DaqNodeSetAction]:
        if self._api is not None:
            await set_parallel(self._api, nodes)
            return []
        return [
            DaqNodeSetAction(
                self.daq,
                node.path,
                node.value,
                caching_strategy=CachingStrategy.CACHE
                if node.cache
                else CachingStrategy.NO_CACHE,
                filename=node.filename,
            )
            for node in nodes.set_actions()
        ]

    async def maybe_async_wait(self, nodes: dict[str, Any]) -> dict[str, Any]:
        if self._api is not None:
            rw = ResponseWaiter()
            rw.add(target=self, conditions=nodes)
            await rw.wait_all(timeout=1)
            return {}
        return nodes

    def clear_cache(self):
        if self._api is not None:
            # TODO(2K): the code below is only needed to keep async API behavior
            # in emulation mode matching that of legacy API with L1Q cache.
            if isinstance(self._api, MockInstrument):
                self._api.clear_cache()
        else:
            self.daq.clear_cache()

    def add_command_table_header(self, body: dict) -> dict:
        # Stub, implement in sub-class
        _logger.debug("Command table unavailable on device %s", self.dev_repr)
        return {}

    def command_table_path(self, awg_index: int) -> str:
        # Stub, implement in sub-class
        _logger.debug("No command table available for device %s", self.dev_repr)
        return ""

    def _warn_for_unsupported_param(self, param_assert, param_name, channel):
        if not param_assert:
            channel_clause = (
                "" if channel is None else f" specified for the channel {channel}"
            )
            _logger.warning(
                "%s: parameter '%s'%s is not supported on this device type.",
                self.dev_repr,
                param_name,
                channel_clause,
            )

    def _check_expected_dev_opts(self):
        if self.is_secondary:
            return
        actual_opts = "/".join([self.dev_type, *self.dev_opts]).upper()
        if self.options.expected_installed_options is None:
            _logger.warning(
                f"{self.dev_repr}: Include the device options '{actual_opts}' in the"
                f" device setup ('options' field of the 'instruments' list in the device"
                f" setup descriptor). This will become a strict requirement in the future."
            )
        elif actual_opts != self.options.expected_installed_options.upper():
            _logger.warning(
                f"{self.dev_repr}: The expected device options specified in the device"
                f" setup '{self.options.expected_installed_options}' do not match the"
                f" actual options '{actual_opts}'. Currently using the actual options,"
                f" but please note that exact matching will become a strict"
                f" requirement in the future."
            )

    def _process_dev_opts(self):
        pass

    def _get_sequencer_type(self) -> str:
        return "auto"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        return SequencerPaths(
            elf=f"/{self.serial}/awgs/{index}/elf/data",
            progress=f"/{self.serial}/awgs/{index}/elf/progress",
            enable=f"/{self.serial}/awgs/{index}/enable",
            ready=f"/{self.serial}/awgs/{index}/ready",
        )

    def add_downlink(self, port: str, linked_device_uid: str, linked_device: DeviceZI):
        self._downlinks.setdefault(port, []).append(
            (linked_device_uid, ref(linked_device))
        )

    def add_uplink(self, linked_device: DeviceZI):
        dev_ref = ref(linked_device)
        if dev_ref not in self._uplinks:
            self._uplinks.append(dev_ref)

    def remove_all_links(self):
        self._downlinks.clear()
        self._uplinks.clear()

    def downlinks(self) -> Iterator[tuple[str, str, DeviceZI]]:
        for port, downstream_devices in self._downlinks.items():
            for uid, dev_ref in downstream_devices:
                yield port, uid, dev_ref()

    def is_leader(self):
        # Check also downlinks, to exclude standalone devices
        return len(self._uplinks) == 0 and len(self._downlinks) > 0

    def is_follower(self):
        # Treat standalone devices as followers
        return len(self._uplinks) > 0 or self.is_standalone()

    def is_standalone(self):
        def is_ppc(dev):
            return (
                getattr(getattr(dev, "device_qualifier", None), "driver", None)
                == "SHFPPC"
            )

        no_ppc_uplinks = [u for u in self._uplinks if u() and not is_ppc(u())]
        return len(no_ppc_uplinks) == 0 and len(self._downlinks) == 0

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        outputs = initialization.outputs or []
        for output in outputs:
            yield DeviceAttribute(
                name=AttributeName.OUTPUT_SCHEDULER_PORT_DELAY,
                index=output.channel,
                value_or_param=output.scheduler_port_delay,
            )
            yield DeviceAttribute(
                name=AttributeName.OUTPUT_PORT_DELAY,
                index=output.channel,
                value_or_param=output.port_delay,
            )

        inputs = initialization.inputs or []
        for input in inputs:
            yield DeviceAttribute(
                name=AttributeName.INPUT_SCHEDULER_PORT_DELAY,
                index=input.channel,
                value_or_param=input.scheduler_port_delay,
            )
            yield DeviceAttribute(
                name=AttributeName.INPUT_PORT_DELAY,
                index=input.channel,
                value_or_param=input.port_delay,
            )

    def validate_scheduled_experiment(
        self, device_uid: str, scheduled_experiment: ScheduledExperiment
    ):
        pass

    async def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        return []

    async def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        return []

    async def _connect_to_data_server(
        self, emulator_state: EmulatorState | None, use_async_api: bool
    ):
        if self._connected:
            return

        _logger.debug("%s: Connecting to %s interface.", self.dev_repr, self.interface)
        try:
            if use_async_api:
                self._api = await create_device_kernel_session(
                    device_qualifier=self._device_qualifier,
                    server_qualifier=self.daq.server_qualifier,
                    emulator_state=emulator_state,
                )
                self._node_monitor = NodeMonitorAsync(self._api)
            else:
                self.daq.connectDevice(self.serial, self.interface)
                self._node_monitor = self.daq.node_monitor
        except RuntimeError as exc:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Connecting failed"
            ) from exc

        _logger.debug("%s: Connected to %s interface.", self.dev_repr, self.interface)

        dev_type_path = f"/{self.serial}/features/devtype"
        dev_opts_path = f"/{self.serial}/features/options"
        dev_traits = await self.get_raw_values(f"{dev_type_path},{dev_opts_path}")
        dev_type = dev_traits.get(dev_type_path)
        dev_opts = dev_traits.get(dev_opts_path)
        if isinstance(dev_type, str):
            self.dev_type = dev_type
        if isinstance(dev_opts, str):
            self.dev_opts = dev_opts.split("\n")
        self._process_dev_opts()

        self._connected = True

    async def connect(self, emulator_state: EmulatorState | None, use_async_api: bool):
        await self._connect_to_data_server(emulator_state, use_async_api=use_async_api)
        if self._node_monitor is not None:
            self.node_monitor.add_nodes(self.nodes_to_monitor())

    def disconnect(self):
        if not self._connected:
            return

        if self._api is None:
            self.daq.disconnectDevice(self.serial)
        else:
            self._api = None  # TODO(2K): Proper disconnect?
        self._connected = False

    async def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        """Returns actions to disable the specified outputs for the device.

        outputs: set(int)
            - When 'invert' is False: set of outputs to disable.
            - When 'invert' is True: set of used outputs to be skipped, remaining
            outputs will be disabled.

        invert: bool
            Controls how 'outputs' argument is interpreted, see above. Special case: set
            to True along with empty 'outputs' to disable all outputs.
        """
        return []

    def on_experiment_end(self) -> NodeCollector:
        return NodeCollector()

    def free_allocations(self):
        self._allocated_oscs.clear()
        self._allocated_awgs.clear()
        self._pipeliner_reload_tracker.clear()

    def _nodes_to_monitor_impl(self):
        nodes = []
        nodes.extend([node.path for node in self.load_factory_preset_control_nodes()])
        nodes.extend([node.path for node in self.clock_source_control_nodes()])
        nodes.extend([node.path for node in self.system_freq_control_nodes()])
        nodes.extend([node.path for node in self.rf_offset_control_nodes()])
        nodes.extend([node.path for node in self.zsync_link_control_nodes()])
        nodes.extend(self.collect_warning_nodes())
        return nodes

    def update_clock_source(self, force_internal: bool | None):
        pass

    def update_rf_offsets(self, rf_offsets: dict[int, float]):
        self._rf_offsets = rf_offsets

    def load_factory_preset_control_nodes(self) -> list[NodeControlBase]:
        return []

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        return []

    def system_freq_control_nodes(self) -> list[NodeControlBase]:
        return []

    def rf_offset_control_nodes(self) -> list[NodeControlBase]:
        return []

    def zsync_link_control_nodes(self) -> list[NodeControlBase]:
        return []

    def nodes_to_monitor(self) -> list[str]:
        if self._nodes_to_monitor is None:
            self._nodes_to_monitor = self._nodes_to_monitor_impl()
        return self._nodes_to_monitor

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self,
        osc_group_oscs: list[AllocatedOscillator],
        osc_param: OscillatorParam,
        recipe_data: RecipeData,
    ) -> int | None:
        return None

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/oscs/{index}/freq"

    def allocate_osc(self, osc_param: OscillatorParam, recipe_data: RecipeData):
        osc_group = self._osc_group_by_channel(osc_param.channel)
        osc_group_oscs = [o for o in self._allocated_oscs if o.group == osc_group]
        same_id_osc = next((o for o in osc_group_oscs if o.id == osc_param.id), None)
        if same_id_osc is None:
            # pylint: disable=E1128
            new_index = self._get_next_osc_index(osc_group_oscs, osc_param, recipe_data)
            if new_index is None:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: exceeded the number of available oscillators for "
                    f"channel {osc_param.channel}"
                )
            self._allocated_oscs.append(
                AllocatedOscillator(
                    group=osc_group,
                    channels={osc_param.channel},
                    index=new_index,
                    id=osc_param.id,
                    frequency=osc_param.frequency,
                    param=osc_param.param,
                )
            )
        else:
            if same_id_osc.frequency != osc_param.frequency:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: ambiguous frequency in recipe for oscillator "
                    f"'{osc_param.id}': {same_id_osc.frequency} != {osc_param.frequency}"
                )
            same_id_osc.channels.add(osc_param.channel)

    async def configure_feedback(
        self, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        return []

    async def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        with_pipeliner: bool,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        return []

    async def get_raw(self, path: str) -> dict[str, Any]:
        if self._api is not None:
            return await get_raw(self._api, path)
        return self.daq.get_raw(path)

    async def get_raw_values(self, path: str) -> dict[str, Any]:
        return {p: v["value"][-1] for p, v in (await self.get_raw(path)).items()}

    async def get_measurement_data(
        self,
        recipe_data: RecipeData,
        channel: int,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ) -> RawReadoutData:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support result retrieval"
        )

    async def get_input_monitor_data(
        self, channel: int, num_results: int
    ) -> RawReadoutData | None:
        return None  # default -> no results available from the device

    async def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {}

    async def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {}

    async def collect_execution_setup_nodes(
        self, with_pipeliner: bool, has_awg_in_use: bool
    ) -> list[DaqNodeSetAction]:
        return []

    async def collect_execution_teardown_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeSetAction]:
        return []

    async def conditions_for_sync_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {}

    async def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        pass

    def _adjust_frequency(self, freq):
        return freq

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector()
        for osc in self._allocated_oscs:
            osc_index = recipe_data.oscillator_ids.index(osc.id)
            [osc_freq], updated = attributes.resolve(
                keys=[(AttributeName.OSCILLATOR_FREQ, osc_index)]
            )
            if updated:
                osc_freq_adjusted = self._adjust_frequency(osc_freq)
                for ch in osc.channels:
                    nc.add(self._make_osc_path(ch, osc.index), osc_freq_adjusted)
        return nc

    def _choose_wf_collector(
        self, elf_nodes: NodeCollector, wf_nodes: NodeCollector
    ) -> NodeCollector:
        return elf_nodes

    def _elf_upload_condition(self, awg_index: int) -> dict[str, Any]:
        return {}

    async def prepare_artifacts(
        self,
        recipe_data: RecipeData,
        rt_section_uid: str,
        initialization: Initialization,
        awg_index: int,
        nt_step: NtStepKey,
    ) -> tuple[
        DeviceZI, list[DaqNodeSetAction], list[DaqNodeSetAction], dict[str, Any]
    ]:
        # TODO(2K): Interface need to be more generic, abstract away DaqNodeSetAction
        artifacts = recipe_data.get_artifacts(self._device_class, ArtifactsCodegen)
        rt_execution_info = recipe_data.rt_execution_infos[rt_section_uid]

        if rt_execution_info.with_pipeliner and not self.has_pipeliner:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Pipeliner is not supported by the device."
            )

        elf_nodes = NodeCollector()
        wf_nodes = NodeCollector()
        wf_eff = self._choose_wf_collector(elf_nodes, wf_nodes)
        upload_ready_conditions: dict[str, Any] = {}

        if rt_execution_info.with_pipeliner:
            elf_nodes.extend(self.pipeliner_prepare_for_upload(awg_index))

        for pipeliner_job in range(rt_execution_info.pipeliner_jobs):
            effective_nt_step = (
                NtStepKey(indices=tuple([*nt_step.indices, pipeliner_job]))
                if rt_execution_info.with_pipeliner
                else nt_step
            )
            rt_exec_step = next(
                (
                    r
                    for r in recipe_data.recipe.realtime_execution_init
                    if r.device_id == initialization.device_uid
                    and r.awg_id == awg_index
                    and r.nt_step == effective_nt_step
                ),
                None,
            )

            if rt_execution_info.with_pipeliner:
                rt_exec_step = self._pipeliner_reload_tracker[awg_index].calc_next_step(
                    pipeliner_job=pipeliner_job,
                    rt_exec_step=rt_exec_step,
                )

            if rt_exec_step is None:
                continue

            seqc_code = self.prepare_seqc(artifacts, rt_exec_step.seqc_ref)
            if seqc_code is not None:
                seqc_item = SeqCCompileItem(
                    dev_type=self.dev_type,
                    dev_opts=self.dev_opts,
                    awg_index=awg_index,
                    sequencer=self._get_sequencer_type(),
                    sampling_rate=self._sampling_rate,
                    code=seqc_code,
                    filename=rt_exec_step.seqc_ref,
                )
                await seqc_compile_async(seqc_item)
                elf_nodes.extend(
                    self.prepare_upload_elf(
                        seqc_item.elf, awg_index, seqc_item.filename
                    )
                )
                upload_ready_conditions.update(self._elf_upload_condition(awg_index))

            waves = self.prepare_waves(artifacts, rt_exec_step.wave_indices_ref)
            if waves is not None:
                acquisition_type = RtExecutionInfo.get_acquisition_type_def(
                    rt_execution_info
                )
                wf_eff.extend(
                    self.prepare_upload_all_binary_waves(
                        awg_index, waves, acquisition_type
                    )
                )

            command_table = self.prepare_command_table(
                artifacts, rt_exec_step.wave_indices_ref
            )
            if command_table is not None:
                wf_eff.extend(
                    self.prepare_upload_command_table(awg_index, command_table)
                )

            wf_eff.extend(
                # TODO(2K): Cleanup arguments to prepare_upload_all_integration_weights
                self.prepare_upload_all_integration_weights(
                    recipe_data,
                    initialization.device_uid,
                    awg_index,
                    artifacts,
                    recipe_data.recipe.integrator_allocations,
                    rt_exec_step.kernel_indices_ref,
                )
            )

            wf_eff.extend(
                self.prepare_pipeliner_job_nodes(
                    recipe_data,
                    rt_section_uid,
                    AwgKey(initialization.device_uid, awg_index),
                    pipeliner_job,
                )
            )

            if rt_execution_info.with_pipeliner:
                # For devices with pipeliner, wf_eff == elf_nodes
                wf_eff.extend(self.pipeliner_commit(awg_index))

        if rt_execution_info.with_pipeliner:
            upload_ready_conditions.update(self.pipeliner_ready_conditions(awg_index))

        elf_nodes_actions = await self.maybe_async(elf_nodes)
        upload_ready_conditions = await self.maybe_async_wait(upload_ready_conditions)
        wf_nodes_actions = await self.maybe_async(wf_nodes)

        return self, elf_nodes_actions, wf_nodes_actions, upload_ready_conditions

    @staticmethod
    def _contains_only_zero_or_one(a):
        if a is None:
            return True
        return not np.any(a * (1 - a))

    def _prepare_markers_iq(
        self, waves: dict[str, CodegenWaveform], sig: str
    ) -> npt.NDArray[Any] | None:
        marker1_wave = get_wave(f"{sig}_marker1.wave", waves, optional=True)
        marker2_wave = get_wave(f"{sig}_marker2.wave", waves, optional=True)

        marker_samples = None
        if marker1_wave is not None:
            if not self._contains_only_zero_or_one(marker1_wave.samples):
                raise LabOneQControllerException(
                    "Marker samples must only contain ones and zeros"
                )
            marker_samples = np.array(marker1_wave.samples, order="C")
        if marker2_wave is not None:
            marker2_len = len(marker2_wave.samples)
            if marker_samples is None:
                marker_samples = np.zeros(marker2_len, dtype=np.int32)
            elif len(marker_samples) != marker2_len:
                raise LabOneQControllerException(
                    "Samples for marker1 and marker2 must have the same length"
                )
            if not self._contains_only_zero_or_one(marker2_wave.samples):
                raise LabOneQControllerException(
                    "Marker samples must only contain ones and zeros"
                )
            # we want marker 2 to be played on output 2, marker 1
            # bits 0/1 = marker 1/2 of output 1, bit 2/4 = marker 1/2 output 2
            # bit 2 is factor 4
            factor = 4
            marker_samples += factor * np.asarray(marker2_wave.samples, order="C")
        return marker_samples

    def _prepare_markers_single(
        self, waves: dict[str, CodegenWaveform], sig: str
    ) -> npt.NDArray[Any] | None:
        marker_wave = get_wave(f"{sig}_marker1.wave", waves, optional=True)

        if marker_wave is None:
            return None
        if not self._contains_only_zero_or_one(marker_wave.samples):
            raise LabOneQControllerException(
                "Marker samples must only contain ones and zeros"
            )
        return np.array(marker_wave.samples, order="C")

    def _prepare_wave_iq(
        self, waves: dict[str, CodegenWaveform], sig: str, index: int
    ) -> WaveformItem:
        wave_i = get_wave(f"{sig}_i.wave", waves)
        wave_q = get_wave(f"{sig}_q.wave", waves)
        marker_samples = self._prepare_markers_iq(waves, sig)
        return WaveformItem(
            index=index,
            name=sig,
            samples=zhinst.utils.convert_awg_waveform(
                np.clip(np.ascontiguousarray(wave_i.samples), -1, 1),
                np.clip(np.ascontiguousarray(wave_q.samples), -1, 1),
                markers=marker_samples,
            ),
        )

    def _prepare_wave_single(
        self, waves: dict[str, CodegenWaveform], sig: str, index: int
    ) -> WaveformItem:
        wave = get_wave(f"{sig}.wave", waves)
        marker_samples = self._prepare_markers_single(waves, sig)
        return WaveformItem(
            index=index,
            name=sig,
            samples=zhinst.utils.convert_awg_waveform(
                np.clip(np.ascontiguousarray(wave.samples), -1, 1),
                markers=marker_samples,
            ),
        )

    def _prepare_wave_complex(
        self, waves: dict[str, CodegenWaveform], sig: str, index: int
    ) -> WaveformItem:
        wave = get_wave(f"{sig}.wave", waves)
        return WaveformItem(
            index=index,
            name=sig,
            samples=np.ascontiguousarray(wave.samples, dtype=np.complex128),
            hold_start=wave.hold_start,
            hold_length=wave.hold_length,
        )

    def prepare_waves(
        self,
        artifacts: ArtifactsCodegen,
        wave_indices_ref: str | None,
    ) -> Waveforms | None:
        if wave_indices_ref is None:
            return None
        wave_indices: dict[str, list[int | str]] = next(
            (i for i in artifacts.wave_indices if i["filename"] == wave_indices_ref),
            {"value": {}},
        )["value"]

        waves: Waveforms = []
        index: int
        sig_type: str
        for sig, [index, sig_type] in wave_indices.items():
            if sig.startswith("precomp_reset"):
                continue  # precomp reset waveform is bundled with ELF
            if sig_type in ("iq", "double", "multi"):
                wave = self._prepare_wave_iq(artifacts.waves, sig, index)
            elif sig_type == "single":
                wave = self._prepare_wave_single(artifacts.waves, sig, index)
            elif sig_type == "complex":
                wave = self._prepare_wave_complex(artifacts.waves, sig, index)
            else:
                raise LabOneQControllerException(
                    f"Unexpected signal type for binary wave for '{sig}' in '{wave_indices_ref}' - "
                    f"'{sig_type}', should be one of [iq, double, multi, single, complex]"
                )
            waves.append(wave)

        return waves

    def prepare_command_table(
        self,
        artifacts: ArtifactsCodegen,
        ct_ref: str | None,
    ) -> dict | None:
        if ct_ref is None:
            return None

        command_table_body = next(
            (ct["ct"] for ct in artifacts.command_tables if ct["seqc"] == ct_ref),
            None,
        )

        if command_table_body is None:
            return None

        oscillator_map = {osc.id: osc.index for osc in self._allocated_oscs}
        command_table_body = deepcopy(command_table_body)
        for entry in command_table_body:
            if "oscillatorSelect" not in entry:
                continue
            oscillator_uid = entry["oscillatorSelect"]["value"]["$ref"]
            entry["oscillatorSelect"]["value"] = oscillator_map[oscillator_uid]

        return self.add_command_table_header(command_table_body)

    def prepare_seqc(
        self,
        artifacts: ArtifactsCodegen,
        seqc_ref: str | None,
    ) -> str | None:
        if seqc_ref is None:
            return None

        seqc = next((s for s in artifacts.src if s["filename"] == seqc_ref), None)
        if seqc is None:
            raise LabOneQControllerException(f"SeqC program '{seqc_ref}' not found")

        # Substitute oscillator nodes by actual assignment
        seqc_lines = seqc["text"].split("\n")
        for i, seqc_line in enumerate(seqc_lines):
            m = seqc_osc_match.match(seqc_line)
            if m is not None:
                param = m.group(2)
                for osc in self._allocated_oscs:
                    if osc.param == param:
                        seqc_lines[i] = (
                            f"{m.group(1)}{m.group(2)}{m.group(3)}{osc.index}{m.group(4)}"
                        )

        # Substitute oscillator index by actual assignment
        for osc in self._allocated_oscs:
            osc_index_symbol = string_sanitize(osc.id)
            pattern = re.compile(rf"const {osc_index_symbol} = \w+;")
            for i, l in enumerate(seqc_lines):
                if not pattern.match(l):
                    continue
                seqc_lines[i] = f"const {osc_index_symbol} = {osc.index};  // final"

        seqc_text = "\n".join(seqc_lines)

        return seqc_text

    def prepare_upload_elf(
        self, elf: bytes, awg_index: int, filename: str
    ) -> NodeCollector:
        nc = NodeCollector()
        sequencer_paths = self.get_sequencer_paths(awg_index)
        nc.add(
            sequencer_paths.elf,
            elf,
            cache=False,
            filename=filename,
        )
        return nc

    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: npt.ArrayLike,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        nc = NodeCollector()
        nc.add(
            f"/{self.serial}/awgs/{awg_index}/waveform/waves/{wave_index}",
            waveform,
            cache=False,
            filename=filename,
        )
        return nc

    def prepare_upload_all_binary_waves(
        self,
        awg_index,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        # Default implementation for "old" devices, override for newer devices
        nc = NodeCollector()
        for wave in waves:
            nc.extend(
                self.prepare_upload_binary_wave(
                    filename=wave.name,
                    waveform=wave.samples,
                    awg_index=awg_index,
                    wave_index=wave.index,
                    acquisition_type=acquisition_type,
                )
            )
        return nc

    def prepare_upload_command_table(
        self, awg_index, command_table: dict
    ) -> NodeCollector:
        command_table_path = self.command_table_path(awg_index)
        nc = NodeCollector()
        nc.add(
            command_table_path + "data",
            json.dumps(command_table, sort_keys=True),
            cache=False,
        )
        return nc

    def prepare_upload_all_integration_weights(
        self,
        recipe_data: RecipeData,
        device_uid: str,
        awg_index: int,
        artifacts: ArtifactsCodegen,
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str,
    ) -> NodeCollector:
        return NodeCollector()

    def prepare_pipeliner_job_nodes(
        self,
        recipe_data: RecipeData,
        rt_section_uid: str,
        awg_key: AwgKey,
        pipeliner_job: int,
    ) -> NodeCollector:
        return NodeCollector()

    def pipeliner_prepare_for_upload(self, index: int) -> NodeCollector:
        return []

    def pipeliner_commit(self, index: int) -> NodeCollector:
        return []

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        return {}

    def _get_num_awgs(self) -> int:
        return 0

    async def collect_osc_initialization_nodes(self) -> list[DaqNodeSetAction]:
        nc = NodeCollector()
        osc_inits = {
            self._make_osc_path(ch, osc.index): osc.frequency
            for osc in self._allocated_oscs
            for ch in osc.channels
        }
        for path, freq in osc_inits.items():
            nc.add(path, 0 if freq is None else self._adjust_frequency(freq))
        return await self.maybe_async(nc)

    async def collect_awg_before_upload_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        return []

    async def collect_awg_after_upload_nodes(
        self, initialization: Initialization
    ) -> list[DaqNodeSetAction]:
        return []

    async def collect_execution_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        _logger.debug("%s: Executing AWGS...", self.dev_repr)

        for awg_index in self._allocated_awgs:
            _logger.debug("%s: Starting AWG #%d sequencer", self.dev_repr, awg_index)
            nc.add(f"awgs/{awg_index}/enable", 1, cache=False)

        return await self.maybe_async(nc)

    async def collect_internal_start_execution_nodes(self) -> list[DaqNodeSetAction]:
        return []

    async def fetch_errors(self):
        error_node = f"/{self.serial}/raw/error/json/errors"
        all_errors = await self.get_raw(error_node)
        return all_errors[error_node]

    async def collect_reset_nodes(self) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("raw/error/clear", 1, cache=False)
        return await self.maybe_async(nc)

    def collect_warning_nodes(self) -> list[str]:
        return []

    def update_warning_nodes(self, node_values: dict[str, Any]):
        pass
