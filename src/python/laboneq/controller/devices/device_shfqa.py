# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio

import logging
from typing import Any, Iterator

from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.qachannel import QAChannel
import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import _gather
from laboneq.controller.devices.device_shf_base import DeviceSHFBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    SequencerPaths,
    RawReadoutData,
)
from laboneq.controller.recipe_processor import (
    RecipeData,
    RtExecutionInfo,
    WaveformItem,
    Waveforms,
    get_artifacts,
    get_initialization_by_device_uid,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.controller.utilities.for_each import for_each
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.data.recipe import (
    IO,
    Initialization,
    IntegratorAllocation,
    NtStepKey,
    TriggeringMode,
)
from laboneq.data.scheduled_experiment import ArtifactsCodegen, ScheduledExperiment


_logger = logging.getLogger(__name__)

INTERNAL_TRIGGER_CHANNEL = 8  # PQSC style triggering on the SHFSG/QC
SOFTWARE_TRIGGER_CHANNEL = 1024  # Software triggering on the SHFQA

SAMPLE_FREQUENCY_HZ = 2.0e9


def _integrator_has_consistent_msd_num_state(
    integrator_allocation: IntegratorAllocation,
):
    num_states = integrator_allocation.kernel_count + 1
    num_thresholds = len(integrator_allocation.thresholds)
    num_expected_thresholds = (num_states - 1) * num_states // 2

    def pluralize(n, noun):
        return f"{n} {noun}{'s' if n > 1 else ''}"

    if num_thresholds != num_expected_thresholds:
        raise LabOneQControllerException(
            f"Multi discrimination configuration of experiment is not consistent."
            f" For {pluralize(num_states, 'state')}, I expected"
            f" {pluralize(integrator_allocation.kernel_count, 'kernel')}"
            f" and {pluralize(num_expected_thresholds, 'threshold')}, but got"
            f" {pluralize(num_thresholds, 'threshold')}.\n"
            "For n states, there should be n-1 kernels, and (n-1)*n/2 thresholds."
        )


class DeviceSHFQA(DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFQA4"
        self.dev_opts = []
        self._qachannels: list[QAChannel] = []
        self._channels = 4
        self._integrators = 16
        self._long_readout_available = True
        self._wait_for_awgs = True
        self._emit_trigger = False

    @property
    def has_pipeliner(self) -> bool:
        return True

    def all_channels(self) -> Iterator[ChannelBase]:
        return iter(self._qachannels)

    def allocated_channels(self) -> Iterator[ChannelBase]:
        for ch in self._allocated_awgs:
            yield self._qachannels[ch]

    def pipeliner_prepare_for_upload(self, index: int) -> NodeCollector:
        return self._qachannels[index].pipeliner.prepare_for_upload()

    def pipeliner_commit(self, index: int) -> NodeCollector:
        return self._qachannels[index].pipeliner.commit()

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        return self._qachannels[index].pipeliner.ready_conditions()

    @property
    def dev_repr(self) -> str:
        if self.options.is_qc:
            return f"SHFQC/QA:{self.serial}"
        return f"SHFQA:{self.serial}"

    def _process_dev_opts(self):
        self._check_expected_dev_opts()
        self._process_shf_opts()
        if self.dev_type == "SHFQA4":
            self._channels = 4
        elif self.dev_type == "SHFQA2":
            self._channels = 2
        elif self.dev_type == "SHFQC":
            self._channels = 1
        else:
            _logger.warning(
                "%s: Unknown device type '%s', assuming SHFQA4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4
        if "16W" in self.dev_opts or self.dev_type == "SHFQA4":
            self._integrators = 16
        else:
            self._integrators = 8
        self._long_readout_available = "LRT" in self.dev_opts
        self._qachannels = [
            QAChannel(
                api=self._api,
                subscriber=self._subscriber,
                device_uid=self.uid,
                serial=self.serial,
                channel=ch,
                integrators=self._integrators,
                repr_base=self.dev_repr,
                is_plus=self._is_plus,
            )
            for ch in range(self._channels)
        ]

    def _get_sequencer_type(self) -> str:
        return "qa"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        qachannel = self._qachannels[index]
        return SequencerPaths(
            elf=qachannel.nodes.generator_elf_data,
            progress=qachannel.nodes.generator_elf_progress,
            enable=qachannel.nodes.generator_enable,
            ready=qachannel.nodes.generator_ready,
        )

    def _validate_range(self, io: IO, is_out: bool):
        if io.range is None:
            return
        input_ranges = np.array(
            [-50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 5, 10],
            dtype=np.float64,
        )
        output_ranges = np.array(
            [-30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=np.float64
        )
        range_list = output_ranges if is_out else input_ranges
        label = "Output" if is_out else "Input"

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
        artifacts = get_artifacts(scheduled_experiment.artifacts, ArtifactsCodegen)
        long_readout_signals = artifacts.requires_long_readout.get(self.uid, [])
        if len(long_readout_signals) > 0:
            if not self._long_readout_available:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Experiment requires long readout that is not available on the device."
                )

        initialization = get_initialization_by_device_uid(
            scheduled_experiment.recipe, self.uid
        )
        if initialization is not None:
            for output in initialization.outputs:
                self._warn_for_unsupported_param(
                    param_assert=output.offset is None or output.offset == 0,
                    param_name="voltage_offsets",
                    channel=output.channel,
                )
                self._warn_for_unsupported_param(
                    param_assert=output.gains is None,
                    param_name="correction_matrix",
                    channel=output.channel,
                )
                if output.range is not None:
                    self._validate_range(output, is_out=True)
                if output.enable_output_mute and not self._is_plus:
                    _logger.warning(
                        f"{self.dev_repr}: Device output muting is enabled, but the device is not"
                        " SHF+ and therefore no muting will happen. It is suggested to disable it."
                    )

            for input in initialization.inputs:
                self._validate_range(input, is_out=False)
                matching_output: IO | None = next(
                    (
                        output
                        for output in initialization.outputs
                        if output.channel == input.channel
                    ),
                    None,
                )
                if matching_output is None:
                    continue
                assert input.port_mode == matching_output.port_mode, (
                    f"{self.dev_repr}: Port mode mismatch between input and output of"
                    f" channel {input.channel}."
                )

    def validate_recipe_data(self, recipe_data: RecipeData):
        for integrator_allocation in recipe_data.recipe.integrator_allocations:
            _integrator_has_consistent_msd_num_state(integrator_allocation)

    def _make_osc_path(self, channel: int, index: int) -> str:
        return self._qachannels[channel].nodes.osc_freq[index]

    async def disable_outputs(self, outputs: set[int], invert: bool):
        await for_each(
            self.all_channels(),
            ChannelBase.disable_output,
            outputs=outputs,
            invert=invert,
        )

    def _result_node_scope(self, ch: int) -> str:
        return f"/{self.serial}/scopes/0/channels/{ch}/wave"

    def _busy_nodes(self) -> list[str]:
        if not self._setup_caps.supports_shf_busy:
            return []
        return [self._qachannels[ch].nodes.busy for ch in self._allocated_awgs]

    def configure_acquisition(
        self,
        recipe_data: RecipeData,
        awg_index: int,
        pipeliner_job: int,
    ) -> NodeCollector:
        return self._qachannels[awg_index].configure_acquisition(
            recipe_data=recipe_data, pipeliner_job=pipeliner_job
        )

    async def start_execution(self, with_pipeliner: bool):
        await for_each(
            self.allocated_channels(),
            ChannelBase.start_execution,
            with_pipeliner=with_pipeliner,
        )

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, nt_step: NtStepKey, with_pipeliner: bool
    ):
        hw_sync = with_pipeliner and (
            self._has_awg_in_use(recipe_data) or self.options.is_qc
        )
        nc = NodeCollector(base=f"/{self.serial}/")
        if hw_sync and self._emit_trigger:
            nc.add("system/internaltrigger/synchronization/enable", 1)  # enable
        if hw_sync and not self._emit_trigger:
            nc.add("system/synchronization/source", 1)  # external
        await self.set_async(nc)

    async def emit_start_trigger(self, with_pipeliner: bool):
        if self._emit_trigger:
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add(
                (
                    "system/internaltrigger/enable"
                    if self.options.is_qc
                    else "system/swtriggers/0/single"
                ),
                1,
                cache=False,
            )
            await self.set_async(nc)

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in self._allocated_awgs:
            if with_pipeliner:
                conditions.update(
                    self._qachannels[
                        awg_index
                    ].pipeliner.conditions_for_execution_ready()
                )
            else:
                # TODO(janl): Not sure whether we need this condition on the SHFQA (including SHFQC)
                # as well. The state of the generator enable wasn't always picked up reliably, so we
                # only check in cases where we rely on external triggering mechanisms.
                conditions[self.get_sequencer_paths(awg_index).enable] = (
                    1,
                    f"Readout pulse generator {awg_index} didn't start.",
                )
        return conditions

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in self._allocated_awgs:
            if with_pipeliner:
                conditions.update(
                    self._qachannels[
                        awg_index
                    ].pipeliner.conditions_for_execution_done()
                )
            else:
                conditions[self.get_sequencer_paths(awg_index).enable] = (
                    0,
                    f"Generator {awg_index} didn't stop. Missing start trigger? Check ZSync.",
                )
        return conditions

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        if not self.is_standalone():
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            await self.set_async(
                NodeCollector.one(f"/{self.serial}/system/synchronization/source", 0)
            )

        await for_each(
            self.allocated_channels(),
            QAChannel.teardown_one_step_execution,
            with_pipeliner=with_pipeliner,
        )

    def pre_process_attributes(
        self,
        initialization: Initialization,
    ) -> Iterator[DeviceAttribute]:
        yield from super().pre_process_attributes(initialization)

        for output in initialization.outputs or []:
            if output.amplitude is not None:
                yield DeviceAttribute(
                    name=AttributeName.QA_OUT_AMPLITUDE,
                    index=output.channel,
                    value_or_param=output.amplitude,
                )

        center_frequencies: dict[int, int] = {}
        ios = (initialization.outputs or []) + (initialization.inputs or [])
        for idx, io in enumerate(ios):
            if io.lo_frequency is not None:
                if io.channel in center_frequencies:
                    prev_io_idx = center_frequencies[io.channel]
                    if ios[prev_io_idx].lo_frequency != io.lo_frequency:
                        raise LabOneQControllerException(
                            f"{self.dev_repr}: Local oscillator frequency mismatch between IOs "
                            f"sharing channel {io.channel}: "
                            f"{ios[prev_io_idx].lo_frequency} != {io.lo_frequency}"
                        )
                    continue
                center_frequencies[io.channel] = idx
                yield DeviceAttribute(
                    name=AttributeName.QA_CENTER_FREQ,
                    index=io.channel,
                    value_or_param=io.lo_frequency,
                )

    async def apply_initialization(self, recipe_data: RecipeData):
        device_recipe_data = recipe_data.device_settings.get(self.uid)
        if device_recipe_data is None:
            return

        await for_each(
            self.all_channels(),
            QAChannel.apply_initialization,
            device_recipe_data=device_recipe_data,
        )

    async def _set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        await for_each(
            self.all_channels(),
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
        if is_spectroscopy(acquisition_type):
            return self._qachannels[awg_index].upload_spectroscopy_envelope(wave)
        return self._qachannels[awg_index].upload_generator_wave(wave)

    def prepare_upload_all_binary_waves(
        self,
        awg_index: int,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        return self._qachannels[awg_index].prepare_upload_all_binary_waves(
            waves, acquisition_type
        )

    def prepare_upload_all_integration_weights(
        self,
        recipe_data: RecipeData,
        awg_index: int,
        artifacts: ArtifactsCodegen,
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str | None,
    ) -> NodeCollector:
        return self._qachannels[awg_index].prepare_upload_all_integration_weights(
            recipe_data,
            artifacts,
            integrator_allocations,
            kernel_ref,
        )

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        await for_each(
            self.all_channels(),
            QAChannel.set_before_awg_upload,
            recipe_data=recipe_data,
        )

    async def configure_trigger(self, recipe_data: RecipeData):
        _logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        initialization = recipe_data.get_initialization(self.uid)
        triggering_mode = initialization.config.triggering_mode

        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            pass
        elif triggering_mode == TriggeringMode.DESKTOP_LEADER:
            self._wait_for_awgs = False
            self._emit_trigger = True
            if self.options.is_qc:
                nc = NodeCollector(base=f"/{self.serial}/")
                nc.add("system/internaltrigger/enable", 0)
                nc.add("system/internaltrigger/repetitions", 1)
                await self.set_async(nc)
        else:
            raise LabOneQControllerException(
                f"Unsupported triggering mode: {triggering_mode} for device type SHFQA."
            )

        trig_channel = 0
        if initialization.config.triggering_mode == TriggeringMode.DESKTOP_LEADER:
            # standalone QA oder QC
            trig_channel = (
                INTERNAL_TRIGGER_CHANNEL
                if self.options.is_qc
                else SOFTWARE_TRIGGER_CHANNEL
            )

        await for_each(
            self.allocated_channels(),
            QAChannel.configure_trigger,
            trig_channel=trig_channel,
        )
        if len(self._allocated_awgs) == 0:
            # TODO(2K): Not clear what this workaround is addressing, original code:
            # https://gitlab.zhinst.com/qccs/qccs/-/merge_requests/1303/diffs#6ac44cb93a82be7127f076e9108cf97cf6c8f4b5_809_810
            nc = NodeCollector(base=f"/{self.serial}/qachannels/0/")
            nc.add("markers/0/source", 32)
            nc.add("markers/1/source", 36)
            await self.set_async(nc)

    async def on_experiment_begin(self):
        subscribe_qa_nodes = [
            self._qachannels[ch].subscribe_nodes() for ch in self._allocated_awgs
        ]
        subscribe_scope_nodes = NodeCollector()
        for ch in self._allocated_awgs:
            subscribe_scope_nodes.add_path(self._result_node_scope(ch))

        nodes = NodeCollector.all([*subscribe_qa_nodes, subscribe_scope_nodes])
        await _gather(
            super().on_experiment_begin(),
            *(self._subscriber.subscribe(self._api, path) for path in nodes.paths()),
        )

    async def on_experiment_end(self):
        await super().on_experiment_end()
        nc = NodeCollector(base=f"/{self.serial}/")
        # in CW spectroscopy mode, turn off the tone
        nc.add("qachannels/*/spectroscopy/envelope/enable", 1, cache=False)
        await self.set_async(nc)

    async def get_measurement_data(
        self,
        channel: int,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ) -> RawReadoutData:
        return await self._qachannels[channel].get_measurement_data(
            rt_execution_info=rt_execution_info,
            result_indices=result_indices,
            num_results=num_results,
        )

    def _ch_repr_scope(self, ch: int) -> str:
        return f"{self.dev_repr}:scope:ch{ch}"

    async def get_raw_data(
        self, channel: int, acquire_length: int, acquires: int | None
    ) -> RawReadoutData:
        result_path = self._result_node_scope(channel)
        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5.0
        # Segment lengths are always multiples of 16 samples.
        segment_length = (acquire_length + 0xF) & (~0xF)
        if acquires is None:
            acquires = 1
        try:
            raw_result = await self._subscriber.get_result(
                result_path, timeout_s=timeout_s
            )
            raw_data = np.reshape(raw_result.vector, (acquires, segment_length))
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{self._ch_repr_scope(channel)}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            raw_data = np.full((acquires, segment_length), np.nan, dtype=np.complex128)
        return RawReadoutData(raw_data)

    async def reset_to_idle(self):
        await super().reset_to_idle()
        nc = NodeCollector(base=f"/{self.serial}/")
        # Reset pipeliner first, attempt to set generator enable leads to FW error if pipeliner was enabled.
        nc.add("qachannels/*/pipeliner/reset", 1, cache=False)
        nc.add("qachannels/*/pipeliner/mode", 0, cache=False)  # off
        nc.add("qachannels/*/synchronization/enable", 0, cache=False)
        nc.barrier()
        nc.add("qachannels/*/generator/enable", 0, cache=False)
        nc.add("system/synchronization/source", 0, cache=False)  # internal
        if self.options.is_qc:
            nc.add("system/internaltrigger/synchronization/enable", 0, cache=False)
        nc.add("qachannels/*/readout/result/enable", 0, cache=False)
        nc.add("qachannels/*/spectroscopy/psd/enable", 0, cache=False)
        nc.add("qachannels/*/spectroscopy/result/enable", 0, cache=False)
        nc.add("qachannels/*/output/rflfinterlock", 1, cache=False)
        if self._long_readout_available:
            nc.add("qachannels/*/modulation/enable", 0, cache=False)
            nc.add("qachannels/*/generator/waveforms/*/hold/enable", 0, cache=False)
            nc.add(
                "qachannels/*/readout/integration/downsampling/factor", 1, cache=False
            )
        # Factory value after reset is 0.5 to avoid clipping during interpolation.
        # We set it to 1.0 for consistency with integration mode.
        nc.add("qachannels/*/oscs/*/gain", 1.0, cache=False)
        nc.add("scopes/0/enable", 0, cache=False)
        nc.add("scopes/0/channels/*/enable", 0, cache=False)
        await self.set_async(nc)

    def _collect_warning_nodes(self) -> list[tuple[str, str]]:
        warning_nodes = []
        for channel in self.all_channels():
            warning_nodes.extend(channel.collect_warning_nodes())
        return warning_nodes
