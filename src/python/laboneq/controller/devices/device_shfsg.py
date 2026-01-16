# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Iterator

from laboneq.controller.attribute_value_tracker import (
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.core_base import CoreBase, SHFChannelBase
from laboneq.controller.devices.device_shf_base import (
    OPT_OUTPUT_ROUTER_ADDER,
    DeviceSHFBase,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.node_control import NodeControlBase, Setting
from laboneq.controller.devices.sgchannel import SGChannel, SHFSGMixIn
from laboneq.controller.recipe_processor import (
    AwgType,
    DeviceRecipeData,
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
)
from laboneq.controller.utilities.for_each import for_each
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.recipe import (
    Initialization,
    NtStepKey,
)
from laboneq.data.scheduled_experiment import ScheduledExperiment

_logger = logging.getLogger(__name__)


class DeviceSHFSG(SHFSGMixIn, DeviceSHFBase):
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
        self._emit_trigger = False
        self._has_opt_rtr = False

    def all_cores(self) -> Iterable[CoreBase]:
        return iter(self._sgchannels)

    def allocated_cores(self, recipe_data: RecipeData) -> Iterable[CoreBase]:
        for ch in recipe_data.allocated_awgs(self.uid):
            yield self._sgchannels[ch]

    def full_channels(self) -> Iterable[CoreBase]:
        for sgchannel in self._sgchannels:
            if sgchannel.is_full:
                yield sgchannel

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
        self._has_opt_rtr = OPT_OUTPUT_ROUTER_ADDER in self.dev_opts
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
        elif self.dev_type == "SHFSG2":
            self._outputs = 2
            self._channels = self._outputs
            self._output_to_synth_map = [0, 1]
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
                core_index=core_index,
                repr_base=self.dev_repr,
                is_plus=self._is_plus,
                has_opt_rtr=self._has_opt_rtr,
                is_full=core_index < self._outputs,
                is_standalone=self.is_standalone(),
            )
            for core_index in range(self._channels)
        ]

    def validate_scheduled_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        rt_execution_info: RtExecutionInfo,
    ):
        self._validate_scheduled_experiment_shfsg(
            scheduled_experiment=scheduled_experiment
        )

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)
        yield from self._pre_process_attributes_shfsg(initialization)

    def _make_osc_path(self, channel: int, index: int, awg_type: AwgType) -> str:
        return self._sgchannels[channel].nodes.osc_freq[index]

    def _busy_nodes(self, recipe_data: RecipeData) -> list[str]:
        if not self._setup_caps.supports_shf_busy:
            return []
        return [
            self._sgchannels[ch].nodes.busy
            for ch in recipe_data.allocated_awgs(self.uid)
        ]

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        if self.is_secondary:
            return []  # QA will initialize the nodes
        else:
            return super().clock_source_control_nodes()

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, nt_step: NtStepKey, with_pipeliner: bool
    ):
        # SG is secondary means it's a part of QC with QA actively used.
        # In this case QA side takes care of synchronization settings.
        # TODO(2K): Cleanup once SHFQC is correctly modelled in the controller.
        if not self.is_secondary:
            hw_sync = with_pipeliner and self._has_awg_in_use(recipe_data)
            await self._set_hw_sync(hw_sync=hw_sync, emit_trigger=self._emit_trigger)

    async def emit_start_trigger(self, recipe_data: RecipeData):
        if self._emit_trigger:
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add("system/internaltrigger/enable", 1, cache=False)
            await self.set_async(nc)

    async def teardown_one_step_execution(self, recipe_data: RecipeData):
        if not self.is_standalone() and not self.is_secondary:
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            await self._api.set_parallel(
                NodeCollector.one(f"/{self.serial}/system/synchronization/source", 0)
            )

        await self._teardown_one_step_execution_shfsg()
        await for_each(
            self.allocated_cores(recipe_data=recipe_data),
            SHFChannelBase.teardown_one_step_execution,
            with_pipeliner=recipe_data.rt_execution_info.is_chunked,
        )

    async def _apply_initialization(self, device_recipe_data: DeviceRecipeData):
        await self._apply_initialization_shfsg(device_recipe_data=device_recipe_data)

    async def _set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        await self._set_nt_step_nodes_shfsg(attributes=attributes)
        await for_each(
            self.full_channels(),
            CoreBase.set_nt_step_nodes,
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
        device_recipe_data = recipe_data.device_settings[self.uid]

        if not device_recipe_data.is_present:
            # Happens for SHFQC/SG when only QA part is configured
            return

        self._emit_trigger = False

        nc = NodeCollector(base=f"/{self.serial}/")

        if device_recipe_data.has_feedback:
            # HACK: HBAR-1427 and HBAR-2165 show that runtime checks generate
            # wrongly detected gaps when enabled during experiments with feedback.
            # Here we ensure that the gap detector is disabled if we are
            # configuring feedback.
            nc.add("raw/system/awg/runtimechecks/enable", 0)

        for awg_index, awg_config in device_recipe_data.awg_configs.items():
            if awg_config.source_feedback_register is None:
                # if it does not have feedback
                continue

            global_feedback = not (
                awg_config.source_feedback_register == "local" and self.is_secondary
            )

            if global_feedback:
                nc.add(f"sgchannels/{awg_index}/awg/diozsyncswitch", 1)  # ZSync Trigger

        if self.is_standalone():  # standalone SHFSG or SHFQC
            if not self.is_secondary:
                # otherwise, the QA will initialize the nodes
                self._emit_trigger = True
                nc.add("system/internaltrigger/enable", 0)
                nc.add("system/internaltrigger/repetitions", 1)
            for awg_index in device_recipe_data.allocated_awgs(default_awg=0):
                nc.add(f"sgchannels/{awg_index}/awg/auxtriggers/0/slope", 1)  # Rise
                nc.add(
                    f"sgchannels/{awg_index}/awg/auxtriggers/0/channel", 8
                )  # Internal trigger

        await self.set_async(nc)

    def command_table_path(self, awg_index: int) -> str:
        return self._sgchannels[awg_index].nodes.awg_command_table + "/"

    async def reset_to_idle(self):
        if not self.is_secondary:
            await super().reset_to_idle()
        nc_sg = SGChannel.reset_to_idle_nodes(
            base=f"/{self.serial}",
            is_qc=self.options.is_qc is True,
            is_secondary=self.is_secondary,
            has_opt_rtr=self._has_opt_rtr,
        )
        await self._api.set_parallel(nc_sg)

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
