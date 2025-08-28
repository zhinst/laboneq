# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any, Iterator

from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.sgchannel import SGChannel
from laboneq.controller.utilities.for_each import for_each
from laboneq.data.scheduled_experiment import ScheduledExperiment
import numpy as np
from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.device_shf_base import (
    DeviceSHFBase,
    check_synth_frequency,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import SequencerPaths
from laboneq.controller.devices.node_control import NodeControlBase, Setting
from laboneq.controller.recipe_processor import (
    RecipeData,
    WaveformItem,
    get_initialization_by_device_uid,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import (
    IO,
    Initialization,
    NtStepKey,
    TriggeringMode,
)


_logger = logging.getLogger(__name__)

OPT_OUTPUT_ROUTER_ADDER = "RTR"


class DeviceSHFSG(DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFSG8"
        self.dev_opts = []
        self._sgchannels: list[SGChannel] = []
        # Available number of full output channels (Front panel outputs).
        self._outputs = 8
        # Available number of output channels (RTR option can extend these with internal channels on certain devices)
        self._channels = self._outputs
        self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        self._wait_for_awgs = True
        self._emit_trigger = False
        self._has_opt_rtr = False

    @property
    def has_pipeliner(self) -> bool:
        return True

    def all_channels(self) -> Iterator[ChannelBase]:
        return iter(self._sgchannels)

    def allocated_channels(self) -> Iterator[ChannelBase]:
        for ch in self._allocated_awgs:
            yield self._sgchannels[ch]

    def full_channels(self) -> Iterator[ChannelBase]:
        for sgchannel in self._sgchannels:
            if sgchannel.is_full:
                yield sgchannel

    def pipeliner_prepare_for_upload(self, index: int) -> NodeCollector:
        return self._sgchannels[index].pipeliner.prepare_for_upload()

    def pipeliner_commit(self, index: int) -> NodeCollector:
        return self._sgchannels[index].pipeliner.commit()

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        return self._sgchannels[index].pipeliner.ready_conditions()

    @property
    def dev_repr(self) -> str:
        if self.options.is_qc:
            return f"SHFQC/SG:{self.serial}"
        return f"SHFSG:{self.serial}"

    @property
    def is_secondary(self) -> bool:
        return self.options.qc_with_qa

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        self._process_shf_opts()
        if OPT_OUTPUT_ROUTER_ADDER in self.dev_opts:
            self._has_opt_rtr = True
        if self.dev_type == "SHFSG8":
            self._outputs = 8
            self._channels = self._outputs
            self._output_to_synth_map = [0, 0, 1, 1, 2, 2, 3, 3]
        elif self.dev_type == "SHFSG4":
            self._outputs = 4
            self._channels = self._outputs
            self._output_to_synth_map = [0, 1, 2, 3]
            if self._has_opt_rtr:
                self._channels = 8
        elif self.dev_type == "SHFQC":
            # Different numbering on SHFQC - index 0 are QA synths
            if "QC2CH" in self.dev_opts:
                self._outputs = 2
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1]
            elif "QC4CH" in self.dev_opts:
                self._outputs = 4
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1, 2, 2]
            elif "QC6CH" in self.dev_opts:
                self._outputs = 6
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1, 2, 2, 3, 3]
            else:
                _logger.warning(
                    "%s: No valid channel option found, installed options: [%s]. "
                    "Assuming 2ch device.",
                    self.dev_repr,
                    ", ".join(self.dev_opts),
                )
                self._outputs = 2
                self._channels = self._outputs
                self._output_to_synth_map = [1, 1]
            if self._has_opt_rtr:
                self._channels = 6
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming SHFSG4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._outputs = 4
            self._channels = self._outputs
            if self._has_opt_rtr:
                self._channels = 8
            self._output_to_synth_map = [0, 1, 2, 3]
        self._sgchannels = [
            SGChannel(
                api=self._api,
                subscriber=self._subscriber,
                device_uid=self.uid,
                serial=self.serial,
                channel=ch,
                repr_base=self.dev_repr,
                is_plus=self._is_plus,
                has_opt_rtr=self._has_opt_rtr,
                is_full=ch < self._outputs,
            )
            for ch in range(self._channels)
        ]

    def _is_full_channel(self, channel: int) -> bool:
        return channel < self._outputs

    def _is_internal_channel(self, channel: int) -> bool:
        return self._outputs < channel < self._channels

    def _get_sequencer_type(self) -> str:
        return "sg"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        sgchannel = self._sgchannels[index]
        return SequencerPaths(
            elf=sgchannel.nodes.awg_elf_data,
            progress=sgchannel.nodes.awg_elf_progress,
            enable=sgchannel.nodes.awg_enable,
            ready=sgchannel.nodes.awg_ready,
        )

    def _validate_range(self, io: IO):
        if io.range is None:
            return
        range_list = np.array([-30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=np.float64)
        label = "Output"

        if io.range_unit not in (None, "dBm"):
            raise LabOneQControllerException(
                f"{label} range of device {self.dev_repr} is specified in "
                f"units of {io.range_unit}. Units must be 'dBm'."
            )
        if not any(np.isclose([io.range] * len(range_list), range_list)):
            _logger.warning(
                "%s: %s channel %d range %.1f is not on the list of allowed ranges: %s. "
                "Nearest allowed range will be used.",
                self.dev_repr,
                label,
                io.channel,
                io.range,
                range_list,
            )

    def validate_scheduled_experiment(self, scheduled_experiment: ScheduledExperiment):
        initialization = get_initialization_by_device_uid(
            scheduled_experiment.recipe, self.uid
        )
        if initialization is None:
            return
        for output in initialization.outputs:
            self._validate_range(output)
            self._warn_for_unsupported_param(
                output.offset is None or output.offset == 0,
                "voltage_offsets",
                output.channel,
            )
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )
            if output.marker_mode not in (None, "TRIGGER", "MARKER"):
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Marker mode must be either 'MARKER' or 'TRIGGER', but got {output.marker_mode} for output {output.channel}"
                )
            if output.enable_output_mute and not self._is_plus:
                _logger.warning(
                    f"{self.dev_repr}: Device output muting is enabled, but the device is not"
                    " SHF+ and therefore no muting will happen. It is suggested to disable it."
                )
            if len(output.routed_outputs) > 0:
                if not self._has_opt_rtr:
                    msg = f"{self.dev_repr}: Output router and adder requires '{OPT_OUTPUT_ROUTER_ADDER}' option on SHFSG / SHFQC devices."
                    raise LabOneQControllerException(msg)
                if not self._is_full_channel(output.channel):
                    msg = f"{self.dev_repr}: Outputs can only be routed to device front panel outputs. Invalid channel: {output.channel}"
                    raise LabOneQControllerException(msg)

    def _make_osc_path(self, channel: int, index: int) -> str:
        return self._sgchannels[channel].nodes.osc_freq[index]

    def _busy_nodes(self) -> list[str]:
        if not self._setup_caps.supports_shf_busy:
            return []
        return [self._sgchannels[ch].nodes.busy for ch in self._allocated_awgs]

    async def disable_outputs(self, outputs: set[int], invert: bool):
        await for_each(
            self.all_channels(),
            ChannelBase.disable_output,
            outputs=outputs,
            invert=invert,
        )

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        if self.is_secondary:
            return []  # QA will initialize the nodes
        else:
            return super().clock_source_control_nodes()

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, nt_step: NtStepKey, with_pipeliner: bool
    ):
        hw_sync = (
            with_pipeliner
            and self._has_awg_in_use(recipe_data)
            and not self.is_secondary
        )
        nc = NodeCollector(base=f"/{self.serial}/")
        if hw_sync and self._emit_trigger:
            nc.add("system/internaltrigger/synchronization/enable", 1)  # enable
        if hw_sync and not self._emit_trigger:
            nc.add("system/synchronization/source", 1)  # external
        await self.set_async(nc)

    async def start_execution(self, with_pipeliner: bool):
        await for_each(
            self.allocated_channels(),
            ChannelBase.start_execution,
            with_pipeliner=with_pipeliner,
        )

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        if not self._wait_for_awgs:
            return {}

        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in self._allocated_awgs:
            if with_pipeliner:
                conditions.update(
                    self._sgchannels[
                        awg_index
                    ].pipeliner.conditions_for_execution_ready()
                )
            else:
                conditions[self.get_sequencer_paths(awg_index).enable] = (
                    1,
                    f"AWG {awg_index} didn't start.",
                )
        return conditions

    async def emit_start_trigger(self, with_pipeliner: bool):
        if self._emit_trigger:
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add("system/internaltrigger/enable", 1, cache=False)
            await self.set_async(nc)

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in self._allocated_awgs:
            if with_pipeliner:
                conditions.update(
                    self._sgchannels[
                        awg_index
                    ].pipeliner.conditions_for_execution_done()
                )
            else:
                conditions[self.get_sequencer_paths(awg_index).enable] = (
                    0,
                    f"AWG {awg_index} didn't stop. Missing start trigger? Check ZSync.",
                )
        return conditions

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        if not self.is_standalone() and not self.is_secondary:
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            nc.add("system/synchronization/source", 0)

        if with_pipeliner:
            for awg_index in self._allocated_awgs:
                nc.extend(self._sgchannels[awg_index].pipeliner.reset_nodes())

        # HACK: HBAR-1427 and HBAR-2165 show that runtime checks generate
        # wrongly detected gaps when enabled during experiments with feedback.
        # Here we make sure that if they were enabled at `session.connect` we
        # re-enable them in case the previous experiment had feedback.
        nc.add("raw/system/awg/runtimechecks/enable", int(self._enable_runtime_checks))

        await self.set_async(nc)

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)

        center_frequencies: dict[int, IO] = {}

        def get_synth_idx(io: IO):
            synth_idx = self._output_to_synth_map[io.channel]
            prev_io = center_frequencies.get(synth_idx)
            if prev_io is None:
                center_frequencies[synth_idx] = io
            elif prev_io.lo_frequency != io.lo_frequency:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Local oscillator frequency mismatch between outputs "
                    f"{prev_io.channel} and {io.channel} sharing synthesizer {synth_idx}: "
                    f"{prev_io.lo_frequency} != {io.lo_frequency}"
                )
            return synth_idx

        ios = initialization.outputs or []
        for io in ios:
            if not self._is_full_channel(io.channel):
                if not self._is_internal_channel(io.channel):
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Attempt to configure channel {io.channel} on a device "
                        f"with {self._outputs} channels. Verify your device setup."
                    )
                continue
            if io.lo_frequency is None:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Local oscillator for channel {io.channel} is required, "
                    f"but is not provided."
                )
            if io.port_mode is None or io.port_mode == "rf":
                yield DeviceAttribute(
                    name=AttributeName.SG_SYNTH_CENTER_FREQ,
                    index=get_synth_idx(io),
                    value_or_param=io.lo_frequency,
                )
            else:
                yield DeviceAttribute(
                    name=AttributeName.SG_DIG_MIXER_CENTER_FREQ,
                    index=io.channel,
                    value_or_param=io.lo_frequency,
                )
            route_ampl_attrs = [
                AttributeName.OUTPUT_ROUTE_1_AMPLITUDE,
                AttributeName.OUTPUT_ROUTE_2_AMPLITUDE,
                AttributeName.OUTPUT_ROUTE_3_AMPLITUDE,
            ]
            route_phase_attrs = [
                AttributeName.OUTPUT_ROUTE_1_PHASE,
                AttributeName.OUTPUT_ROUTE_2_PHASE,
                AttributeName.OUTPUT_ROUTE_3_PHASE,
            ]
            for idx, router in enumerate(io.routed_outputs):
                yield DeviceAttribute(
                    name=route_ampl_attrs[idx],
                    index=io.channel,
                    value_or_param=router.amplitude,
                )
                yield DeviceAttribute(
                    name=route_phase_attrs[idx],
                    index=io.channel,
                    value_or_param=router.phase,
                )

    async def apply_initialization(self, recipe_data: RecipeData):
        device_recipe_data = recipe_data.device_settings.get(self.uid)
        if device_recipe_data is None:
            return

        await for_each(
            self.all_channels(),
            SGChannel.apply_initialization,
            device_recipe_data=device_recipe_data,
        )

        # If multiple oscillators are assigned to a channel, it indicates oscillator switching
        # via the command table, and the oscselect node is ignored. Therefore it can be set to
        # any oscillator.
        nc = NodeCollector(base=f"/{self.serial}/")
        osc_selects = {
            ch: osc.index
            for osc in device_recipe_data.allocated_oscs
            for ch in osc.channels
        }
        for ch, osc_idx in osc_selects.items():
            nc.extend(self._sgchannels[ch]._collect_configure_oscillator_nodes(osc_idx))
        await self.set_async(nc)

    async def _set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        nc = NodeCollector(base=f"/{self.serial}/")
        for synth_idx in set(self._output_to_synth_map):
            [synth_cf], synth_cf_updated = attributes.resolve(
                keys=[(AttributeName.SG_SYNTH_CENTER_FREQ, synth_idx)]
            )
            if synth_cf_updated:
                check_synth_frequency(synth_cf, self.dev_repr, synth_idx)
                nc.add(f"synthesizers/{synth_idx}/centerfreq", synth_cf)
        await self._api.set_parallel(nc)

        await for_each(
            self.full_channels(),
            ChannelBase.set_nt_step_nodes,
            recipe_data=recipe_data,
            attributes=attributes,
        )

    def prepare_upload_binary_wave(
        self,
        awg_index: int,
        wave: WaveformItem,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        return NodeCollector.one(
            path=f"/{self.serial}/sgchannels/{awg_index}/awg/waveform/waves/{wave.index}",
            value=wave.samples,
            cache=False,
            filename=wave.name,
        )

    async def configure_trigger(self, recipe_data: RecipeData):
        _logger.debug("Configuring triggers...")
        initialization = recipe_data.get_initialization(self.uid)
        if initialization.device_type is None:  # dummy initialization
            # Happens for SHFQC/SG when only QA part is configured
            return

        self._wait_for_awgs = True
        self._emit_trigger = False

        nc = NodeCollector(base=f"/{self.serial}/")

        for awg_key, awg_config in recipe_data.awg_configs.items():
            if awg_key.device_uid != self.uid:
                continue

            if awg_config.source_feedback_register is None:
                # if it does not have feedback
                continue

            # HACK: HBAR-1427 and HBAR-2165 show that runtime checks generate
            # wrongly detected gaps when enabled during experiments with feedback.
            # Here we ensure that the gap detector is disabled if we are
            # configuring feedback.
            nc.add("raw/system/awg/runtimechecks/enable", 0)

            global_feedback = not (
                awg_config.source_feedback_register == "local" and self.is_secondary
            )

            if global_feedback:
                nc.add(
                    f"sgchannels/{awg_key.awg_index}/awg/diozsyncswitch", 1
                )  # ZSync Trigger

        triggering_mode = initialization.config.triggering_mode
        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            pass
        elif triggering_mode in (
            TriggeringMode.DESKTOP_LEADER,
            TriggeringMode.INTERNAL_FOLLOWER,
        ):
            # standalone SHFSG or SHFQC
            self._wait_for_awgs = False
            if not self.is_secondary:
                # otherwise, the QA will initialize the nodes
                self._emit_trigger = True
                nc.add("system/internaltrigger/enable", 0)
                nc.add("system/internaltrigger/repetitions", 1)
            for awg_index in (
                self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
            ):
                nc.add(f"sgchannels/{awg_index}/awg/auxtriggers/0/slope", 1)  # Rise
                nc.add(
                    f"sgchannels/{awg_index}/awg/auxtriggers/0/channel", 8
                )  # Internal trigger
        else:
            raise LabOneQControllerException(
                f"Unsupported triggering mode: {triggering_mode} for device type SHFSG."
            )

        await self.set_async(nc)

    def command_table_path(self, awg_index: int) -> str:
        return f"/{self.serial}/sgchannels/{awg_index}/awg/commandtable/"

    async def reset_to_idle(self):
        if not self.is_secondary:
            await super().reset_to_idle()
        nc = NodeCollector(base=f"/{self.serial}/")
        # Reset pipeliner first, attempt to set AWG enable leads to FW error if pipeliner was enabled.
        nc.add("sgchannels/*/pipeliner/reset", 1, cache=False)
        nc.add("sgchannels/*/pipeliner/mode", 0, cache=False)  # off
        nc.add("sgchannels/*/synchronization/enable", 0, cache=False)
        nc.barrier()
        nc.add("sgchannels/*/awg/enable", 0, cache=False)
        if not self.is_secondary:
            nc.add(
                "system/synchronization/source",
                0,  # internal
                cache=False,
            )
            if self.options.is_qc:
                nc.add("system/internaltrigger/synchronization/enable", 0, cache=False)
        if self._has_opt_rtr:
            # Disable any previously configured output routers to make sure they
            # do not introduce signal delay or unexpected signal paths.
            nc.add("sgchannels/*/outputrouter/enable", 0, cache=False)
            nc.add("sgchannels/*/outputrouter/routes/*/enable", 0, cache=False)
        await self.set_async(nc)

    def _collect_warning_nodes(self) -> list[tuple[str, str]]:
        warning_nodes = []
        for channel in self.full_channels():
            warning_nodes.extend(channel.collect_warning_nodes())
        return warning_nodes

    def runtime_check_control_nodes(self) -> list[NodeControlBase]:
        # Enable AWG runtime checks which includes the gap detector.
        return [
            Setting(
                f"/{self.serial}/raw/system/awg/runtimechecks/enable",
                int(self._enable_runtime_checks),
            )
        ]
