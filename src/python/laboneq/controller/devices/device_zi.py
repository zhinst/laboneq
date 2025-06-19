# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator, cast
from weakref import ReferenceType, ref

from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.results import build_partial_result, build_raw_partial_result
from laboneq.controller.utilities.for_each import for_each, for_each_sync
from laboneq.data.experiment_results import ExperimentResults

from laboneq.controller.attribute_value_tracker import (  # pylint: disable=E0401
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    ConditionsCheckerAsync,
    InstrumentConnection,
    ResponseWaiterAsync,
    _gather,
)
from laboneq.controller.devices.device_setup_dao import (
    DeviceOptions,
    ServerQualifier,
    DeviceQualifier,
    DeviceSetupDAO,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.zi_emulator import EmulatorState
from laboneq.controller.devices.node_control import (
    NodeControlBase,
    filter_commands,
    filter_responses,
    filter_settings,
    filter_states,
    filter_wait_conditions,
)
from laboneq.controller.pipeliner_reload_tracker import PipelinerReloadTracker
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
    Waveforms,
    get_elf,
    prepare_command_table,
    prepare_waves,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.versioning import SetupCaps
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import (
    Initialization,
    IntegratorAllocation,
    NtStepKey,
    OscillatorParam,
)
from laboneq.data.scheduled_experiment import (
    ArtifactsCodegen,
    ScheduledExperiment,
)

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray


_logger = logging.getLogger(__name__)


class AwgCompilerStatus(Enum):
    SUCCESS = 0
    ERROR = 1
    WARNING = 2


@dataclass
class AllocatedOscillator:
    # Oscillators may be grouped in HW, restricting certain channels to oscillators
    # from specific groups. Allocation is performed within each group.
    group: int
    channels: set[int]
    index: int
    id: str
    frequency: float | None
    param: str | None


@dataclass
class SequencerPaths:
    elf: str
    progress: str
    enable: str
    ready: str


@dataclass
class RawReadoutData:
    vector: NumPyArray

    # metadata by job_id
    metadata: dict[int, dict[str, Any]] = field(default_factory=dict)


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


class DeviceAbstract(ABC):
    # TODO(2K): This is a dummy abstract base class to make ruff happy. To be removed.
    @abstractmethod
    def _dummy(self): ...


class DeviceZI(DeviceAbstract):
    def __init__(
        self,
        server_qualifier: ServerQualifier,
        device_qualifier: DeviceQualifier,
        setup_caps: SetupCaps,
    ):
        self._server_qualifier = server_qualifier
        self._device_qualifier = device_qualifier
        self._downlinks: list[ReferenceType[DeviceZI]] = []
        self._uplinks: list[ReferenceType[DeviceZI]] = []

    def _dummy(self):
        pass

    @property
    def server_qualifier(self):
        return self._server_qualifier

    @property
    def device_qualifier(self):
        return self._device_qualifier

    @property
    def uid(self) -> str:
        return self.device_qualifier.uid

    @property
    def options(self) -> DeviceOptions:
        return self.device_qualifier.options

    @property
    def serial(self):
        return self.options.serial.lower()

    @property
    def dev_repr(self) -> str:
        return f"{self.device_qualifier.driver.upper()}:{self.serial}"

    @property
    def is_secondary(self) -> bool:
        return False

    ### Device linking by trigger chain
    def remove_all_links(self):
        self._downlinks.clear()
        self._uplinks.clear()

    def add_downlink(self, linked_device: DeviceZI):
        dev_ref = ref(linked_device)
        if dev_ref not in self._downlinks:
            self._downlinks.append(dev_ref)

    def add_uplink(self, linked_device: DeviceZI):
        dev_ref = ref(linked_device)
        if dev_ref not in self._uplinks:
            self._uplinks.append(dev_ref)

    def is_leader(self) -> bool:
        # Check also downlinks, to exclude standalone devices
        return len(self._uplinks) == 0 and len(self._downlinks) > 0

    def is_follower(self) -> bool:
        # Treat standalone devices as followers
        return len(self._uplinks) > 0 or self.is_standalone()

    def is_standalone(self) -> bool:
        return len(self._uplinks) == 0 and len(self._downlinks) == 0

    ### Device setup settings
    def update_clock_source(self, force_internal: bool | None):
        pass

    def update_from_device_setup(self, ds: DeviceSetupDAO):
        pass

    ### Device connectivity
    @abstractmethod
    async def connect(
        self,
        emulator_state: EmulatorState | None,
        disable_runtime_checks: bool,
        timeout_s: float,
    ):
        pass

    @abstractmethod
    def disconnect(self):
        pass

    async def disable_outputs(self, outputs: set[int], invert: bool):
        """Disables the specified outputs for the device.

        outputs: set(int)
            - When 'invert' is False: set of outputs to disable.
            - When 'invert' is True: set of used outputs to be skipped, remaining
            outputs will be disabled.

        invert: bool
            Controls how 'outputs' argument is interpreted, see above. Special case: set
            to True along with empty 'outputs' to disable all outputs.
        """
        pass

    def clear_cache(self):
        pass

    ### Other methods
    async def set_async(self, nodes: NodeCollector):
        pass

    def validate_scheduled_experiment(self, scheduled_experiment: ScheduledExperiment):
        pass

    def validate_recipe_data(self, recipe_data: RecipeData):
        pass

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from []

    @abstractmethod
    def calc_raw_acquire_length(
        self,
        recipe_data: RecipeData,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        signal_id: str,
    ) -> int: ...

    @abstractmethod
    async def prepare_artifacts(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
    ):
        pass

    def prepare_upload_binary_wave(
        self,
        awg_index: int,
        wave: WaveformItem,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        return NodeCollector()

    def prepare_upload_command_table(
        self, awg_index, command_table: dict
    ) -> NodeCollector:
        return NodeCollector()

    async def fetch_errors(self) -> str | list[str]:
        return []

    def _collect_warning_nodes(self) -> list[tuple[str, str]]:
        return []

    ### Result processing
    async def read_results(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        results: ExperimentResults,
    ):
        pass


class DeviceBase(DeviceZI):
    def __init__(
        self,
        server_qualifier: ServerQualifier,
        device_qualifier: DeviceQualifier,
        setup_caps: SetupCaps,
    ):
        super().__init__(server_qualifier, device_qualifier, setup_caps)
        self._setup_caps = setup_caps

        self._api = InstrumentConnection()
        self._subscriber = AsyncSubscriber()
        self.dev_type: str = "UNKNOWN"
        self.dev_opts: list[str] = []
        self._connected = False
        self._voltage_offsets: dict[int, float] = {}
        self._allocated_oscs: list[AllocatedOscillator] = []
        self._allocated_awgs: set[int] = set()
        self._pipeliner_reload_tracker: dict[int, PipelinerReloadTracker] = defaultdict(
            PipelinerReloadTracker
        )
        self._sampling_rate: float | None = None
        self._enable_runtime_checks = True
        self._warning_nodes: dict[str, int] = {}

        if self.serial is None:
            raise LabOneQControllerException(
                "ZI device must be provided with serial number via options"
            )

        if self.interface is None or self.interface == "":
            raise LabOneQControllerException(
                "ZI device must be provided with interface via options"
            )

    @property
    def has_pipeliner(self) -> bool:
        return False

    @property
    def interface(self):
        return self.options.interface.lower()

    def _has_awg_in_use(self, recipe_data: RecipeData):
        initialization = recipe_data.get_initialization(self.uid)
        return len(initialization.awgs) > 0

    async def set_async(self, nodes: NodeCollector | Iterable[NodeCollector]):
        await self._api.set_parallel(NodeCollector.all(nodes))

    def all_channels(self) -> Iterator[ChannelBase]:
        """Iterable over all channels of the device."""
        return iter([])

    def clear_cache(self):
        # TODO(2K): the code below is only needed to keep async API behavior
        # in emulation mode matching that of legacy API with LabOne Q cache.
        self._api.clear_cache()

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
        actual_opts_str = "/".join([self.dev_type, *self.dev_opts])
        if self.options.expected_dev_type is None:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Device options '{actual_opts_str}' is missing in the"
                f" device setup ('options' field of the 'instruments' list in the device"
                f" setup descriptor, 'device_options' argument when constructing"
                f" instrument objects to be added to 'DeviceSetup' instances)."
            )
        elif self.dev_type != self.options.expected_dev_type or set(
            self.dev_opts
        ) != set(self.options.expected_dev_opts):
            expected_opts_str = "/".join(
                [self.options.expected_dev_type, *self.options.expected_dev_opts]
            )
            raise LabOneQControllerException(
                f"{self.dev_repr}: The expected device options specified in the device"
                f" setup '{expected_opts_str}' do not match the"
                f" actual options '{actual_opts_str}'."
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

    def calc_raw_acquire_length(
        self,
        recipe_data: RecipeData,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        signal_id: str,
    ) -> int:
        if signal_id not in awg_config.signal_raw_acquire_lengths:
            # Use default length 4096, in case AWG config is not available
            return 4096
        return awg_config.signal_raw_acquire_lengths[signal_id]

    def _sigout_from_port(self, ports: list[str]) -> int | None:
        return None

    def _add_voltage_offset(self, sigout: int, voltage_offset: float):
        if sigout in self._voltage_offsets:
            if not math.isclose(self._voltage_offsets[sigout], voltage_offset):
                _logger.warning(
                    "Ambiguous 'voltage_offset' for the output %s of device %s: %s != %s, "
                    "will use %s",
                    sigout,
                    self.uid,
                    self._voltage_offsets[sigout],
                    voltage_offset,
                    self._voltage_offsets[sigout],
                )
        else:
            self._voltage_offsets[sigout] = voltage_offset

    def update_from_device_setup(self, ds: DeviceSetupDAO):
        for calib in ds.calibrations(self.uid):
            if DeviceSetupDAO.is_rf(calib) and calib.voltage_offset is not None:
                sigout = self._sigout_from_port(calib.ports)
                if sigout is not None:
                    self._add_voltage_offset(sigout, calib.voltage_offset)

    async def _connect_to_data_server(
        self,
        emulator_state: EmulatorState | None,
        timeout_s: float,
    ):
        if self._connected:
            return

        _logger.debug("%s: Connecting to %s interface.", self.dev_repr, self.interface)
        try:
            self._api = await InstrumentConnection.connect(
                device_qualifier=self.device_qualifier,
                server_qualifier=self.server_qualifier,
                emulator_state=emulator_state,
                timeout_s=timeout_s,
            )
        except RuntimeError as exc:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Connecting failed"
            ) from exc

        _logger.debug("%s: Connected to %s interface.", self.dev_repr, self.interface)

        dev_type_path = f"/{self.serial}/features/devtype"
        dev_opts_path = f"/{self.serial}/features/options"
        dev_traits = await self._api.get_raw([dev_type_path, dev_opts_path])
        dev_type = dev_traits.get(dev_type_path)
        dev_opts = dev_traits.get(dev_opts_path)
        if isinstance(dev_type, str):
            self.dev_type = dev_type.upper()
        if isinstance(dev_opts, str):
            self.dev_opts = dev_opts.upper().splitlines()
        self._process_dev_opts()

        self._connected = True

    async def connect(
        self,
        emulator_state: EmulatorState | None,
        disable_runtime_checks: bool,
        timeout_s: float,
    ):
        self._enable_runtime_checks = not disable_runtime_checks
        await self._connect_to_data_server(emulator_state, timeout_s=timeout_s)

    def disconnect(self):
        if not self._connected:
            return

        self._api = InstrumentConnection()  # TODO(2K): Proper disconnect?
        self._connected = False

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

    def _busy_nodes(self) -> list[str]:
        return []

    def allocate_resources(self, recipe_data: RecipeData):
        self._allocated_oscs.clear()
        self._allocated_awgs.clear()
        self._pipeliner_reload_tracker.clear()
        for_each_sync(self.all_channels(), ChannelBase.allocate_resources)

        device_recipe_data = recipe_data.device_settings[self.uid]
        for osc_param in device_recipe_data.osc_params:
            self._allocate_osc_impl(osc_param, recipe_data)

        for awg_key in recipe_data.awg_configs.keys():
            if awg_key.device_uid == self.uid:
                assert isinstance(awg_key.awg_index, int)
                self._allocated_awgs.add(awg_key.awg_index)

    def _allocate_osc_impl(self, osc_param: OscillatorParam, recipe_data: RecipeData):
        osc_group = self._osc_group_by_channel(osc_param.channel)
        osc_group_oscs = [o for o in self._allocated_oscs if o.group == osc_group]
        same_id_osc = next((o for o in osc_group_oscs if o.id == osc_param.id), None)
        if same_id_osc is None:
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

    async def on_experiment_begin(self):
        await _gather(
            *(
                self._subscriber.subscribe(self._api, path)
                for path, _ in self._collect_warning_nodes()
            ),
            *(
                self._subscriber.subscribe(self._api, path, get_initial=True)
                for path in self._busy_nodes()
            ),
        )

    async def on_experiment_end(self):
        self._subscriber.unsubscribe_all()

    async def init_warning_nodes(self):
        nodes = [node for node, _ in self._collect_warning_nodes()]
        if len(nodes) == 0:
            return
        init_vals = await self._api.get_raw(nodes)
        self._warning_nodes.update(init_vals)

    async def update_warning_nodes(self):
        for node, msg in self._collect_warning_nodes():
            updates = self._subscriber.get_updates(node)
            value = updates[-1].value if len(updates) > 0 else None
            if not isinstance(value, int):
                continue

            prev = self._warning_nodes.get(node)
            if prev is not None and value > prev:
                _logger.warning("%s: %s: %s", self.dev_repr, msg, value - prev)
            self._warning_nodes[node] = value

    def configure_acquisition(
        self,
        recipe_data: RecipeData,
        awg_index: int,
        pipeliner_job: int,
    ) -> NodeCollector:
        return NodeCollector()

    def _adjust_frequency(self, freq):
        return freq

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, user_set_nodes: NodeCollector
    ):
        nc = NodeCollector()

        if not self.is_secondary:
            for node_action in user_set_nodes.set_actions():
                if m := re.match(r"^/?(dev[^/]+)/.+", node_action.path.lower()):
                    serial = m.group(1)
                    if serial == self.serial:
                        nc.add_node_action(node_action)

        attributes = recipe_data.attribute_value_tracker.device_view(self.uid)
        nc.extend(self._collect_prepare_nt_step_nodes(attributes, recipe_data))

        await self.set_async(nc)

    def _collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector()
        for osc in self._allocated_oscs:
            osc_param_index = recipe_data.oscillator_ids.index(osc.id)
            [osc_freq], updated = attributes.resolve(
                keys=[(AttributeName.OSCILLATOR_FREQ, osc_param_index)]
            )
            if updated:
                osc_freq_adjusted = self._adjust_frequency(osc_freq)
                for ch in osc.channels:
                    nc.add(self._make_osc_path(ch, osc.index), osc_freq_adjusted)
        return nc

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        pass

    async def set_after_awg_upload(self, recipe_data: RecipeData):
        pass

    async def configure_feedback(self, recipe_data: RecipeData):
        pass

    async def emit_start_trigger(self, with_pipeliner: bool):
        if self.is_leader():
            await self.start_execution(with_pipeliner=with_pipeliner)

    def _choose_wf_collector(
        self, elf_nodes: NodeCollector, wf_nodes: NodeCollector
    ) -> NodeCollector:
        return elf_nodes

    def _elf_upload_condition(self, awg_index: int) -> dict[str, Any]:
        return {}

    async def prepare_artifacts(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
    ):
        await for_each(self.all_channels(), ChannelBase.load_awg_program)
        await _gather(
            *(
                self._prepare_artifacts_impl(
                    recipe_data=recipe_data,
                    nt_step=nt_step,
                    target_awg_index=awg_index,
                )
                for awg_index in self._allocated_awgs
            )
        )

    async def _prepare_artifacts_impl(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        target_awg_index: int,
    ):
        initialization = recipe_data.get_initialization(self.uid)
        init_awgs = [] if initialization.awgs is None else initialization.awgs
        awg_index: int | None = None
        for awg in init_awgs:
            if isinstance(awg.awg, int) and awg.awg == target_awg_index:
                awg_index = target_awg_index
                break
        if awg_index is None:
            return

        artifacts = recipe_data.get_artifacts(ArtifactsCodegen)
        rt_execution_info = recipe_data.rt_execution_info

        if rt_execution_info.with_pipeliner and not self.has_pipeliner:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Pipeliner is not supported by the device."
            )

        elf_nodes = NodeCollector()
        wf_nodes = NodeCollector()
        wf_eff = self._choose_wf_collector(elf_nodes, wf_nodes)
        upload_ready_conditions: dict[str, Any] = {}

        if rt_execution_info.with_pipeliner:
            # enable pipeliner
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
                    if r.device_id == self.uid
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
            # Todo (PW): This currently needlessly reconfigures the acquisition in every
            #  NT step. The acquisition parameters really are constant across NT steps,
            #  we only care about re-enabling the result logger.
            wf_eff.extend(
                self.configure_acquisition(recipe_data, awg_index, pipeliner_job)
            )

            if rt_exec_step is None:
                continue

            seqc_elf = get_elf(artifacts, rt_exec_step.program_ref)
            if seqc_elf is not None:
                elf_nodes.extend(
                    self.prepare_upload_elf(
                        seqc_elf, awg_index, rt_exec_step.program_ref
                    )
                )
                upload_ready_conditions.update(self._elf_upload_condition(awg_index))

            waves = prepare_waves(artifacts, rt_exec_step.wave_indices_ref)
            if waves is not None:
                acquisition_type = rt_execution_info.acquisition_type
                wf_eff.extend(
                    self.prepare_upload_all_binary_waves(
                        awg_index, waves, acquisition_type
                    )
                )

            command_table = prepare_command_table(
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
                    self.uid,
                    awg_index,
                    artifacts,
                    recipe_data.recipe.integrator_allocations,
                    rt_exec_step.kernel_indices_ref,
                )
            )

            if rt_execution_info.with_pipeliner:
                # For devices with pipeliner, wf_eff == elf_nodes
                wf_eff.extend(self.pipeliner_commit(awg_index))

        if rt_execution_info.with_pipeliner:
            upload_ready_conditions.update(self.pipeliner_ready_conditions(awg_index))

        rw = ResponseWaiterAsync(api=self._api, dev_repr=self.dev_repr, timeout_s=10)
        rw.add_nodes(upload_ready_conditions)
        await rw.prepare()
        await self.set_async(elf_nodes)
        await rw.wait()
        await self.set_async(wf_nodes)

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
        awg_index: int,
        wave: WaveformItem,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        return NodeCollector.one(
            path=f"/{self.serial}/awgs/{awg_index}/waveform/waves/{wave.index}",
            value=wave.samples,
            cache=False,
            filename=wave.name,
        )

    def prepare_upload_all_binary_waves(
        self,
        awg_index: int,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        # Default implementation for "old" devices, override for newer devices
        nc = NodeCollector()
        for wave in waves:
            nc.extend(
                self.prepare_upload_binary_wave(
                    awg_index=awg_index,
                    wave=wave,
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

    def pipeliner_prepare_for_upload(self, index: int) -> NodeCollector:
        return NodeCollector()

    def pipeliner_commit(self, index: int) -> NodeCollector:
        return NodeCollector()

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        return {}

    async def configure_trigger(self, recipe_data: RecipeData):
        pass

    async def apply_initialization(self, recipe_data: RecipeData):
        pass

    async def initialize_oscillators(self):
        nc = NodeCollector()
        osc_inits = {
            self._make_osc_path(ch, osc.index): osc.frequency
            for osc in self._allocated_oscs
            for ch in osc.channels
        }
        for path, freq in osc_inits.items():
            nc.add(path, 0 if freq is None else self._adjust_frequency(freq))
        await self.set_async(nc)

    async def fetch_errors(self) -> str | list[str]:
        error_node = f"/{self.serial}/raw/error/json/errors"
        all_errors = await self._api.get_raw([error_node])
        return all_errors[error_node]

    async def reset_to_idle(self):
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("raw/error/clear", 1, cache=False)
        await self.set_async(nc)

    def load_factory_preset_control_nodes(self) -> list[NodeControlBase]:
        return []

    def runtime_check_control_nodes(self) -> list[NodeControlBase]:
        return []

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        return []

    def system_freq_control_nodes(self) -> list[NodeControlBase]:
        return []

    def rf_offset_control_nodes(self) -> list[NodeControlBase]:
        return []

    async def wait_for_zsync_link(self, timeout_s: float):
        pass

    async def exec_config_step(
        self, control_nodes: list[NodeControlBase], config_name: str, timeout_s: float
    ):
        state_nodes = filter_states(control_nodes)
        state_check = ConditionsCheckerAsync(
            self._api,
            {n.path: n.value for n in state_nodes},
        )
        mismatches = await state_check.check()

        commands = filter_commands(control_nodes)
        responses = filter_responses(control_nodes)
        response_waiter = ResponseWaiterAsync(
            api=self._api, dev_repr=self.dev_repr, timeout_s=timeout_s
        )
        changes_to_apply = []
        if len(commands) > 0:
            # 1a. Has unconditional commands? Use simplified flow.
            changes_to_apply = commands
            response_waiter.add_nodes({n.path: n.value for n in responses})
            if len(mismatches) > 0:
                response_waiter.add_nodes(
                    {n.path: n.value for n in filter_wait_conditions(control_nodes)}
                )
        else:
            # 1b. Is device already in the desired state?
            if len(mismatches) > 0:
                changes_to_apply = filter_settings(control_nodes)
                response_waiter.add_nodes({n.path: n.value for n in responses})
                failed_paths = [path for path, _, _ in mismatches]
                failed_nodes = [n for n in control_nodes if n.path in failed_paths]
                response_waiter.add_nodes(
                    {n.path: n.value for n in failed_nodes},
                )

        # Arm the response waiter (for step 3 below) before applying any changes
        await response_waiter.prepare()

        # 2. Apply any necessary node changes (which may be empty)
        nc = NodeCollector()
        for node in changes_to_apply:
            nc.add(node.path, node.raw_value)
        await self._api.set_parallel(nc)

        # 3. Wait for responses to the changes in step 2 and settling of dependent states
        failed_responses = await response_waiter.wait()
        if len(failed_responses) > 0:
            failures = "\n".join(failed_responses)
            raise LabOneQControllerException(
                f"{self.dev_repr}: Internal error: {config_name} is not complete within {timeout_s}s. "
                f"Not fulfilled:\n{failures}"
            )

        # 4. Recheck all the conditions, as some may have potentially changed as a result of step 2
        final_checks = await state_check.check()
        if len(final_checks) > 0:
            failures = "\n".join(
                [f"{p}: {v}  (expected: {e})" for p, v, e in final_checks]
            )
            raise LabOneQControllerException(
                f"{self.dev_repr}: Internal error: {config_name} failed. "
                f"Errors:\n{failures}"
            )

    async def wait_for_channels_ready(self, timeout_s: float):
        await _gather(
            *(
                self._subscriber.wait_for(path, expected=0, timeout_s=timeout_s)
                for path in self._busy_nodes()
            )
        )

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, with_pipeliner: bool
    ):
        pass

    async def start_execution(self, with_pipeliner: bool):
        pass

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {}

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {}

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        pass

    async def wait_for_execution_ready(self, with_pipeliner: bool, timeout_s: float):
        if not self.is_follower():
            # Can't batch everything together, because PQSC/QHUB needs to start execution after HDs
            # otherwise it can finish before AWGs are started, and the trigger is lost.
            return
        rw = ResponseWaiterAsync(
            api=self._api, dev_repr=self.dev_repr, timeout_s=timeout_s
        )
        rw.add_with_msg(
            nodes=self.conditions_for_execution_ready(with_pipeliner=with_pipeliner),
        )
        await rw.prepare()
        await self.start_execution(with_pipeliner=with_pipeliner)
        failed_nodes = await rw.wait()
        if len(failed_nodes) > 0:
            _logger.warning(
                "Conditions to start RT on followers still not fulfilled after %.1f"
                " seconds, nonetheless trying to continue..."
                "\nNot fulfilled:\n%s",
                timeout_s,
                "\n".join(failed_nodes),
            )

    async def make_waiter_for_execution_done(
        self,
        acquisition_type: AcquisitionType,
        with_pipeliner: bool,
        timeout_s: float,
    ):
        response_waiter = ResponseWaiterAsync(
            api=self._api, dev_repr=self.dev_repr, timeout_s=timeout_s
        )
        response_waiter.add_with_msg(
            nodes=self.conditions_for_execution_done(
                acquisition_type=acquisition_type,
                with_pipeliner=with_pipeliner,
            ),
        )
        await response_waiter.prepare()
        return response_waiter

    async def wait_for_execution_done(
        self,
        response_waiter: ResponseWaiterAsync,
        timeout_s: float,
        min_wait_time: float,
    ):
        failed_nodes = await response_waiter.wait()
        if len(failed_nodes) > 0:
            _logger.warning(
                (
                    "Stop conditions still not fulfilled after %f s, estimated"
                    " execution time was %.2f s. Continuing to the next step."
                    "\nNot fulfilled:\n%s"
                ),
                timeout_s,
                min_wait_time,
                "\n".join(failed_nodes),
            )

    async def read_results(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        results: ExperimentResults,
    ):
        rt_execution_info = recipe_data.rt_execution_info
        await _gather(
            *(
                self._read_one_awg_results(
                    recipe_data,
                    nt_step,
                    rt_execution_info,
                    results,
                    awg_key,
                    awg_config,
                )
                for awg_key, awg_config in recipe_data.awgs_producing_results()
                if awg_key.device_uid == self.uid
            )
        )

    async def _read_one_awg_results(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        rt_execution_info: RtExecutionInfo,
        results: ExperimentResults,
        awg_key: AwgKey,
        awg_config: AwgConfig,
    ):
        if rt_execution_info.acquisition_type == AcquisitionType.RAW:
            assert awg_config.raw_acquire_length is not None
            raw_results = await self.get_raw_data(
                channel=cast(int, awg_key.awg_index),
                acquire_length=awg_config.raw_acquire_length,
                acquires=awg_config.result_length,
            )
            # Raw data is per physical port, and is the same for all logical signals of the AWG
            for signal in awg_config.acquire_signals:
                mapping = rt_execution_info.signal_result_map.get(signal, [])
                signal_acquire_length = awg_config.signal_raw_acquire_lengths.get(
                    signal
                )
                if signal_acquire_length is None:
                    continue
                unique_handles = set(mapping)
                for handle in unique_handles:
                    if handle is None:
                        continue  # Ignore unused acquire signal if any
                    result = results.acquired_results[handle]
                    build_raw_partial_result(
                        result=result,
                        nt_step=nt_step,
                        raw_segments=raw_results.vector,
                        result_length=signal_acquire_length,
                        mapping=mapping,
                        handle=handle,
                    )
        else:
            await _gather(
                *(
                    self._read_one_signal_result(
                        recipe_data,
                        nt_step,
                        rt_execution_info,
                        results,
                        awg_key,
                        awg_config,
                        signal,
                        rt_execution_info.effective_averages,
                    )
                    for signal in awg_config.acquire_signals
                )
            )

    async def _read_one_signal_result(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
        rt_execution_info: RtExecutionInfo,
        results: ExperimentResults,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        signal: str,
        effective_averages: int,
    ):
        integrator_allocation = next(
            (
                i
                for i in recipe_data.recipe.integrator_allocations
                if i.signal_id == signal
            ),
            None,
        )
        if integrator_allocation is None:
            return
        assert integrator_allocation.device_id == awg_key.device_uid
        assert integrator_allocation.awg == awg_key.awg_index
        assert awg_config.result_length is not None, "AWG not producing results"
        raw_readout = await self.get_measurement_data(
            recipe_data,
            cast(int, awg_key.awg_index),
            rt_execution_info,
            integrator_allocation.channels,
            awg_config.result_length,
            effective_averages,
        )
        mapping = rt_execution_info.signal_result_map.get(signal, [])
        unique_handles = set(mapping)
        for handle in unique_handles:
            if handle is None:
                continue  # unused entries in sparse result vector map to None handle
            result = results.acquired_results[handle]
            build_partial_result(result, nt_step, raw_readout.vector, mapping, handle)

        timestamps = results.pipeline_jobs_timestamps.setdefault(signal, [])

        for job_id, v in raw_readout.metadata.items():
            # make sure the list is long enough for this job id
            timestamps.extend([float("nan")] * (job_id - len(timestamps) + 1))
            timestamps[job_id] = v["timestamp"]

    async def get_raw_data(
        self, channel: int, acquire_length: int, acquires: int | None
    ) -> RawReadoutData:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support result retrieval"
        )

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
