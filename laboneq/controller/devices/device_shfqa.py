# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio

import itertools
import logging
import time
from typing import Any, Iterator

import numpy as np
from numpy import typing as npt

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttribute,
    DeviceAttributesView,
)
from laboneq.controller.communication import (
    DaqNodeSetAction,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.device_shf_base import DeviceSHFBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    SequencerPaths,
    Waveforms,
    delay_to_rounded_samples,
    IntegrationWeightItem,
    IntegrationWeights,
)
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    DeviceRecipeData,
    RecipeData,
    RtExecutionInfo,
    get_wave,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType, is_spectroscopy
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.calibration import PortMode
from laboneq.data.recipe import (
    IO,
    Initialization,
    IntegratorAllocation,
    Measurement,
    TriggeringMode,
)
from laboneq.data.scheduled_experiment import CompilerArtifact, ArtifactsCodegen

_logger = logging.getLogger(__name__)

INTERNAL_TRIGGER_CHANNEL = 1024  # PQSC style triggering on the SHFSG/QC
SOFTWARE_TRIGGER_CHANNEL = 8  # Software triggering on the SHFQA

SAMPLE_FREQUENCY_HZ = 2.0e9
DELAY_NODE_GRANULARITY_SAMPLES = 4
DELAY_NODE_GENERATOR_MAX_SAMPLES = round(131.058e-6 * SAMPLE_FREQUENCY_HZ)
DELAY_NODE_READOUT_INTEGRATION_MAX_SAMPLES = round(131.07e-6 * SAMPLE_FREQUENCY_HZ)
DELAY_NODE_SPECTROSCOPY_ENVELOPE_MAX_SAMPLES = round(131.07e-6 * SAMPLE_FREQUENCY_HZ)
DELAY_NODE_SPECTROSCOPY_MAX_SAMPLES = round(131.066e-6 * SAMPLE_FREQUENCY_HZ)


# Offsets to align {integration, spectroscopy, scope} delays with playback
INTEGRATION_DELAY_OFFSET = 212e-9  # LabOne Q calibrated value
SPECTROSCOPY_DELAY_OFFSET = 220e-9  # LabOne Q calibrated value
SCOPE_DELAY_OFFSET = INTEGRATION_DELAY_OFFSET  # Equality tested at FW level

MAX_INTEGRATION_WEIGHT_LENGTH = 4096
MAX_WAVEFORM_LENGTH_INTEGRATION = 4096
MAX_WAVEFORM_LENGTH_SPECTROSCOPY = 65536

MAX_AVERAGES_SCOPE = 1 << 16
MAX_AVERAGES_RESULT_LOGGER = 1 << 17
MAX_RESULT_VECTOR_LENGTH = 1 << 19


def calc_theoretical_assignment_vec(num_weights: int) -> np.ndarray:
    """Calculates the theoretical assignment vector, assuming that
    zhinst.utils.QuditSettings ws used to calculate the weights
    and the first d-1 weights were selected as kernels.

    The theoretical assignment vector is determined by the majority vote
    (winner takes all) principle.

    see zhinst/utils/shfqa/multistate.py
    """
    num_states = num_weights + 1
    weight_indices = list(itertools.combinations(range(num_states), 2))
    assignment_len = 2 ** len(weight_indices)
    assignment_vec = np.zeros(assignment_len, dtype=int)

    for assignment_idx in range(assignment_len):
        state_counts = np.zeros(num_states, dtype=int)
        for weight_idx, weight in enumerate(weight_indices):
            above_threshold = (assignment_idx & (2**weight_idx)) != 0
            state_idx = weight[0] if above_threshold else weight[1]
            state_counts[state_idx] += 1
        winner_state = np.argmax(state_counts)
        assignment_vec[assignment_idx] = winner_state

    return assignment_vec


class DeviceSHFQA(AwgPipeliner, DeviceSHFBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFQA4"
        self.dev_opts = []
        self._channels = 4
        self._integrators = 16
        self._wait_for_awgs = True
        self._emit_trigger = False
        self.pipeliner_set_node_base(f"/{self.serial}/qachannels")

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

    def _get_sequencer_type(self) -> str:
        return "qa"

    def get_sequencer_paths(self, index: int) -> SequencerPaths:
        return SequencerPaths(
            elf=f"/{self.serial}/qachannels/{index}/generator/elf/data",
            progress=f"/{self.serial}/qachannels/{index}/generator/elf/progress",
            enable=f"/{self.serial}/qachannels/{index}/generator/enable",
            ready=f"/{self.serial}/qachannels/{index}/generator/ready",
        )

    def _get_num_awgs(self) -> int:
        return self._channels

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

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self, osc_group: int, previously_allocated: int
    ) -> int | None:
        if previously_allocated >= 1:
            return None
        return previously_allocated

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/qachannels/{channel}/oscs/{index}/freq"

    async def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                nc.add(f"qachannels/{ch}/output/on", 0, cache=False)
        return await self.maybe_async(nc)

    def on_experiment_end(self) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.extend(super().on_experiment_end())
        # in CW spectroscopy mode, turn off the tone
        nc.add("qachannels/*/spectroscopy/envelope/enable", 1, cache=False)
        return nc

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        for awg in range(self._get_num_awgs()):
            nodes.extend(
                [
                    f"/{self.serial}/qachannels/{awg}/generator/enable",
                    f"/{self.serial}/qachannels/{awg}/generator/ready",
                    f"/{self.serial}/qachannels/{awg}/spectroscopy/psd/enable",
                    f"/{self.serial}/qachannels/{awg}/spectroscopy/result/enable",
                    f"/{self.serial}/qachannels/{awg}/spectroscopy/result/data/wave",
                    f"/{self.serial}/qachannels/{awg}/readout/result/enable",
                ]
            )
            for result_index in range(self._integrators):
                nodes.append(
                    f"/{self.serial}/qachannels/{awg}/readout/result/data/{result_index}/wave",
                )
            nodes.extend(self.pipeliner_control_nodes(awg))
        return nodes

    async def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        with_pipeliner: bool,
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector()
        if not with_pipeliner:
            average_mode = 0 if averaging_mode == AveragingMode.CYCLIC else 1
            nc.extend(
                self._configure_readout(
                    acquisition_type,
                    awg_key,
                    awg_config,
                    integrator_allocations,
                    averages,
                    average_mode,
                )
            )
            nc.extend(
                self._configure_spectroscopy(
                    acquisition_type,
                    awg_key.awg_index,
                    awg_config.result_length,
                    averages,
                    average_mode,
                )
            )
        nc.extend(
            self._configure_scope(
                enable=acquisition_type == AcquisitionType.RAW,
                channel=awg_key.awg_index,
                averages=averages,
                acquire_length=awg_config.raw_acquire_length,
            )
        )
        return await self.maybe_async(nc)

    def _configure_readout(
        self,
        acquisition_type: AcquisitionType,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        average_mode: int,
    ) -> NodeCollector:
        enable = acquisition_type in [
            AcquisitionType.INTEGRATION,
            AcquisitionType.DISCRIMINATION,
        ]
        channel = awg_key.awg_index
        nc = NodeCollector(base=f"/{self.serial}/")
        if enable:
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            result_length = awg_config.result_length
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"Number of distinct readouts {result_length} on device {self.dev_repr},"
                    f" channel {channel}, exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )
            nc.add(f"qachannels/{channel}/readout/result/length", result_length)
            nc.add(f"qachannels/{channel}/readout/result/averages", averages)
            nc.add(
                f"qachannels/{channel}/readout/result/source",
                # 1 - result_of_integration
                # 3 - result_of_discrimination
                3 if acquisition_type == AcquisitionType.DISCRIMINATION else 1,
            )
            nc.add(f"qachannels/{channel}/readout/result/mode", average_mode)
            nc.add(f"qachannels/{channel}/readout/result/enable", 0)
            if acquisition_type in [
                AcquisitionType.INTEGRATION,
                AcquisitionType.DISCRIMINATION,
            ]:
                for integrator in integrator_allocations:
                    if (
                        integrator.device_id != awg_key.device_uid
                        or integrator.signal_id not in awg_config.acquire_signals
                    ):
                        continue
                    assert len(integrator.channels) == 1
                    integrator_idx = integrator.channels[0]
                    assert self._integrator_has_consistent_msd_num_state(integrator)
                    for state_i, threshold in enumerate(integrator.thresholds):
                        nc.add(
                            f"qachannels/{channel}/readout/multistate/qudits/{integrator_idx}/thresholds/{state_i}/value",
                            threshold or 0.0,
                        )
        nc.add(f"qachannels/{channel}/readout/result/enable", 1 if enable else 0)
        return nc

    def _configure_spectroscopy(
        self,
        acq_type: AcquisitionType,
        channel: int,
        result_length: int,
        averages: int,
        average_mode: int,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        if is_spectroscopy(acq_type):
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            if result_length > MAX_RESULT_VECTOR_LENGTH:
                raise LabOneQControllerException(
                    f"Number of distinct readouts {result_length} on device {self.dev_repr},"
                    f" channel {channel}, exceeds the allowed maximum {MAX_RESULT_VECTOR_LENGTH}"
                )
            nc.add(f"qachannels/{channel}/spectroscopy/result/length", result_length)
            nc.add(f"qachannels/{channel}/spectroscopy/result/averages", averages)
            nc.add(f"qachannels/{channel}/spectroscopy/result/mode", average_mode)
            nc.add(f"qachannels/{channel}/spectroscopy/psd/enable", 0)
            nc.add(f"qachannels/{channel}/spectroscopy/result/enable", 0)
        if acq_type == AcquisitionType.SPECTROSCOPY_PSD:
            nc.add(f"qachannels/{channel}/spectroscopy/psd/enable", 1)
        nc.add(
            f"qachannels/{channel}/spectroscopy/result/enable",
            1 if is_spectroscopy(acq_type) else 0,
        )
        return nc

    def _configure_scope(
        self, enable: bool, channel: int, averages: int, acquire_length: int
    ) -> NodeCollector:
        # TODO(2K): multiple acquire events
        nc = NodeCollector(base=f"/{self.serial}/")
        if enable:
            if averages > MAX_AVERAGES_SCOPE:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_SCOPE}"
                )
            nc.add("scopes/0/time", 0)  # 0 -> 2 GSa/s
            nc.add("scopes/0/averaging/enable", 1)
            nc.add("scopes/0/averaging/count", averages)
            nc.add(f"scopes/0/channels/{channel}/enable", 1)
            nc.add(
                f"scopes/0/channels/{channel}/inputselect", channel
            )  # channelN_signal_input
            nc.add("scopes/0/length", acquire_length)
            nc.add("scopes/0/segments/enable", 0)
            # TODO(2K): multiple acquire events per monitor
            # "scopes/0/segments/enable", 1
            # "scopes/0/segments/count", measurement.result_length
            # TODO(2K): only one trigger is possible for all channels. Which one to use?
            nc.add(
                "scopes/0/trigger/channel", 64 + channel
            )  # channelN_sequencer_monitor0
            nc.add("scopes/0/trigger/enable", 1)
            nc.add("scopes/0/enable", 0)
            nc.add("scopes/0/single", 1, cache=False)
        nc.add("scopes/0/enable", 1 if enable else 0)
        return nc

    async def collect_execution_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeSetAction]:
        if with_pipeliner:
            nc = self.pipeliner_collect_execution_nodes()
        else:
            nc = NodeCollector(base=f"/{self.serial}/")
            for awg_index in self._allocated_awgs:
                nc.add(f"qachannels/{awg_index}/generator/enable", 1, cache=False)
        return await self.maybe_async(nc)

    async def collect_execution_setup_nodes(
        self, with_pipeliner: bool, has_awg_in_use: bool
    ) -> list[DaqNodeSetAction]:
        hw_sync = with_pipeliner and has_awg_in_use
        nc = NodeCollector(base=f"/{self.serial}/")
        if hw_sync and self._emit_trigger:
            nc.add("system/internaltrigger/synchronization/enable", 1)  # enable
        if hw_sync and not self._emit_trigger:
            nc.add("system/synchronization/source", 1)  # external
        return await self.maybe_async(nc)

    async def collect_internal_start_execution_nodes(self) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        if self._emit_trigger:
            nc.add(
                "system/internaltrigger/enable"
                if self.options.is_qc
                else "system/swtriggers/0/single",
                1,
                cache=False,
            )
        return await self.maybe_async(nc)

    async def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, Any]:
        if with_pipeliner:
            conditions = self.pipeliner_conditions_for_execution_ready()
        else:
            # TODO(janl): Not sure whether we need this condition on the SHFQA (including SHFQC)
            # as well. The state of the generator enable wasn't always picked up reliably, so we
            # only check in cases where we rely on external triggering mechanisms.
            conditions = {
                f"/{self.serial}/qachannels/{awg_index}/generator/enable": 1
                for awg_index in self._allocated_awgs
            }
        return await self.maybe_async_wait(conditions)

    async def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, Any]:
        conditions: dict[str, Any] = {}

        if with_pipeliner:
            conditions.update(self.pipeliner_conditions_for_execution_done())
        else:
            conditions.update(
                {
                    f"/{self.serial}/qachannels/{awg_index}/generator/enable": 0
                    for awg_index in self._allocated_awgs
                }
            )
        return await self.maybe_async_wait(conditions)

    async def collect_execution_teardown_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")

        if not self.is_standalone():
            # Deregister this instrument from synchronization via ZSync.
            # HULK-1707: this must happen before disabling the synchronization of the last AWG
            nc.add("system/synchronization/source", 0)

        return await self.maybe_async(nc)

    def _validate_initialization(self, initialization: Initialization):
        super()._validate_initialization(initialization)
        for input in initialization.inputs or []:
            output = next(
                (
                    output
                    for output in initialization.outputs or []
                    if output.channel == input.channel
                ),
                None,
            )
            if output is None:
                continue
            assert input.port_mode == output.port_mode, (
                f"{self.dev_repr}: Port mode mismatch between input and output of"
                f" channel {input.channel}."
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

        center_frequencies = {}
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

    async def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        _logger.debug("%s: Initializing device...", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")

        outputs = initialization.outputs or []
        for output in outputs:
            self._warn_for_unsupported_param(
                output.offset is None or output.offset == 0,
                "voltage_offsets",
                output.channel,
            )
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )
            self._allocated_awgs.add(output.channel)
            nc.add(f"qachannels/{output.channel}/output/on", 1 if output.enable else 0)
            if output.range is not None:
                self._validate_range(output, is_out=True)
                nc.add(f"qachannels/{output.channel}/output/range", output.range)

            nc.add(f"qachannels/{output.channel}/generator/single", 1)
            if self._is_plus:
                nc.add(
                    f"qachannels/{output.channel}/output/muting/enable",
                    int(output.enable_output_mute),
                )
            else:
                if output.enable_output_mute:
                    _logger.warning(
                        f"{self.dev_repr}: Device output muting is enabled, but the device is not"
                        " SHF+ and therefore no mutting will happen. It is suggested to disable it."
                    )
        for input in initialization.inputs or []:
            nc.add(
                f"qachannels/{input.channel}/input/rflfpath",
                1  # RF
                if input.port_mode is None or input.port_mode == PortMode.RF.value
                else 0,  # LF
            )

        return await self.maybe_async(nc)

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.extend(super().collect_prepare_nt_step_nodes(attributes, recipe_data))

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for ch in range(self._channels):
            [synth_cf], synth_cf_updated = attributes.resolve(
                keys=[(AttributeName.QA_CENTER_FREQ, ch)]
            )
            if synth_cf_updated:
                nc.add(f"qachannels/{ch}/centerfreq", synth_cf)

            [out_amp], out_amp_updated = attributes.resolve(
                keys=[(AttributeName.QA_OUT_AMPLITUDE, ch)]
            )
            if out_amp_updated:
                nc.add(f"qachannels/{ch}/oscs/0/gain", out_amp)

            (
                [output_scheduler_port_delay, output_port_delay],
                output_updated,
            ) = attributes.resolve(
                keys=[
                    (AttributeName.OUTPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.OUTPUT_PORT_DELAY, ch),
                ]
            )
            output_delay = (
                0.0
                if output_scheduler_port_delay is None
                else output_scheduler_port_delay + (output_port_delay or 0.0)
            )
            set_output = output_updated and output_scheduler_port_delay is not None

            (
                [input_scheduler_port_delay, input_port_delay],
                input_updated,
            ) = attributes.resolve(
                keys=[
                    (AttributeName.INPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.INPUT_PORT_DELAY, ch),
                ]
            )
            measurement_delay = (
                0.0
                if input_scheduler_port_delay is None
                else input_scheduler_port_delay + (input_port_delay or 0.0)
            )
            set_input = input_updated and input_scheduler_port_delay is not None

            if is_spectroscopy(acquisition_type):
                output_delay_path = f"qachannels/{ch}/spectroscopy/envelope/delay"
                meas_delay_path = f"qachannels/{ch}/spectroscopy/delay"
                measurement_delay += SPECTROSCOPY_DELAY_OFFSET
                max_generator_delay = DELAY_NODE_SPECTROSCOPY_ENVELOPE_MAX_SAMPLES
                max_integrator_delay = DELAY_NODE_SPECTROSCOPY_MAX_SAMPLES
            else:
                output_delay_path = f"qachannels/{ch}/generator/delay"
                meas_delay_path = f"qachannels/{ch}/readout/integration/delay"
                measurement_delay += output_delay
                measurement_delay += (
                    INTEGRATION_DELAY_OFFSET
                    if acquisition_type != AcquisitionType.RAW
                    else SCOPE_DELAY_OFFSET
                )
                set_input = set_input or set_output
                max_generator_delay = DELAY_NODE_GENERATOR_MAX_SAMPLES
                max_integrator_delay = DELAY_NODE_READOUT_INTEGRATION_MAX_SAMPLES

            if set_output:
                output_delay_rounded = (
                    delay_to_rounded_samples(
                        channel=ch,
                        dev_repr=self.dev_repr,
                        delay=output_delay,
                        sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                        granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                        max_node_delay_samples=max_generator_delay,
                    )
                    / SAMPLE_FREQUENCY_HZ
                )
                nc.add(output_delay_path, output_delay_rounded)

            if set_input:
                measurement_delay_rounded = (
                    delay_to_rounded_samples(
                        channel=ch,
                        dev_repr=self.dev_repr,
                        delay=measurement_delay,
                        sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                        granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                        max_node_delay_samples=max_integrator_delay,
                    )
                    / SAMPLE_FREQUENCY_HZ
                )
                if acquisition_type == AcquisitionType.RAW:
                    nc.add("scopes/0/trigger/delay", measurement_delay_rounded)
                nc.add(meas_delay_path, measurement_delay_rounded)

        return nc

    def prepare_integration_weights(
        self,
        artifacts: CompilerArtifact | dict[int, CompilerArtifact],
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str | None,
    ) -> IntegrationWeights | None:
        if isinstance(artifacts, dict):
            artifacts: ArtifactsCodegen = artifacts[self._device_class]
        integration_weights = next(
            (s for s in artifacts.integration_weights if s["filename"] == kernel_ref),
            None,
        )
        if integration_weights is None:
            return

        max_len = MAX_INTEGRATION_WEIGHT_LENGTH

        bin_waves: IntegrationWeights = []
        for signal_id, weight_names in integration_weights["signals"].items():
            integrator_allocation = next(
                ia for ia in integrator_allocations if ia.signal_id == signal_id
            )
            [channel] = integrator_allocation.channels

            for index, weight_name in enumerate(weight_names):
                wave_name = weight_name + ".wave"
                # Note conjugation here:
                weight_vector = np.conjugate(get_wave(wave_name, artifacts.waves))
                wave_len = len(weight_vector)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the integration weight"
                        f" '{channel}' of channel {integrator_allocation.awg} exceeds"
                        f" maximum of {max_len} samples ({max_pulse_len * 1e6:.3f} us)."
                    )
                bin_waves.append(
                    IntegrationWeightItem(
                        integration_unit=channel,
                        index=index,
                        name=wave_name,
                        samples=weight_vector,
                    )
                )

        return bin_waves

    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: npt.ArrayLike,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        assert not is_spectroscopy(acquisition_type) or wave_index == 0
        nc = NodeCollector()
        nc.add(
            f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/wave"
            if is_spectroscopy(acquisition_type)
            else f"/{self.serial}/qachannels/{awg_index}/generator/waveforms/{wave_index}/wave",
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
        nc = NodeCollector()
        has_spectroscopy_envelope = False
        if is_spectroscopy(acquisition_type):
            if len(waves) > 1:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Only one envelope waveform per physical channel is "
                    f"possible in spectroscopy mode. Check play commands for channel {awg_index}."
                )
            max_len = MAX_WAVEFORM_LENGTH_SPECTROSCOPY
            for wave in waves:
                has_spectroscopy_envelope = True
                wave_len = len(wave.samples)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the envelope waveform "
                        f"'{wave.name}' for spectroscopy unit {awg_index} exceeds maximum "
                        f"of {max_len} samples. Ensure measure pulse doesn't "
                        f"exceed {max_pulse_len * 1e6:.3f} us."
                    )
                nc.extend(
                    self.prepare_upload_binary_wave(
                        filename=wave.name,
                        waveform=wave.samples,
                        awg_index=awg_index,
                        wave_index=0,
                        acquisition_type=acquisition_type,
                    )
                )
        else:
            nc.add(
                f"/{self.serial}/qachannels/{awg_index}/generator/clearwave",
                1,
                cache=False,
            )
            max_len = MAX_WAVEFORM_LENGTH_INTEGRATION
            for wave in waves:
                wave_len = len(wave.samples)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the waveform '{wave.name}' "
                        f"for generator {awg_index} / wave slot {wave.index} exceeds maximum "
                        f"of {max_len} samples. Ensure measure pulse doesn't exceed "
                        f"{max_pulse_len * 1e6:.3f} us."
                    )
                nc.extend(
                    self.prepare_upload_binary_wave(
                        filename=wave.name,
                        waveform=wave.samples,
                        awg_index=awg_index,
                        wave_index=wave.index,
                        acquisition_type=acquisition_type,
                    )
                )
        nc.add(
            f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/enable",
            1 if has_spectroscopy_envelope else 0,
        )
        return nc

    def prepare_upload_all_integration_weights(
        self, awg_index, integration_weights: IntegrationWeights
    ) -> NodeCollector:
        nc = NodeCollector(
            base=f"/{self.serial}/qachannels/{awg_index}/readout/multistate/"
        )

        for iw in integration_weights:
            nc.add(
                f"qudits/{iw.integration_unit}/weights/{iw.index}/wave",
                iw.samples,
                filename=iw.name,
            )

        return nc

    def prepare_pipeliner_job_nodes(
        self,
        recipe_data: RecipeData,
        rt_section_uid: str,
        awg_key: AwgKey,
        pipeliner_job: int,
    ) -> NodeCollector:
        nc = NodeCollector()

        rt_execution_info = recipe_data.rt_execution_infos[rt_section_uid]

        if not rt_execution_info.with_pipeliner:
            return nc

        if not recipe_data.setup_caps.result_logger_pipelined and pipeliner_job > 0:
            return nc

        awg_config = recipe_data.awg_configs[awg_key]

        # TODO(2K): code duplication with Controller._prepare_rt_execution
        if rt_execution_info.averaging_mode == AveragingMode.SINGLE_SHOT:
            effective_averages = 1
            effective_averaging_mode = AveragingMode.CYCLIC
            # TODO(2K): handle sequential
        else:
            effective_averages = rt_execution_info.averages
            effective_averaging_mode = rt_execution_info.averaging_mode

        average_mode = 0 if effective_averaging_mode == AveragingMode.CYCLIC else 1
        nc.extend(
            self._configure_readout(
                rt_execution_info.acquisition_type,
                awg_key,
                awg_config,
                recipe_data.recipe.integrator_allocations,
                effective_averages,
                average_mode,
            )
        )
        nc.extend(
            self._configure_spectroscopy(
                rt_execution_info.acquisition_type,
                awg_key.awg_index,
                awg_config.result_length,
                effective_averages,
                average_mode,
            )
        )
        return nc

    def _integrator_has_consistent_msd_num_state(
        self, integrator_allocation: IntegratorAllocation.Data
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
        return True

    def _configure_readout_mode_nodes_multi_state(
        self,
        integrator_allocation: IntegratorAllocation,
        measurement: Measurement,
    ) -> NodeCollector:
        num_states = integrator_allocation.kernel_count + 1
        assert self._integrator_has_consistent_msd_num_state(integrator_allocation)

        assert len(integrator_allocation.channels) == 1, (
            f"{self.dev_repr}: Internal error - expected 1 integrator for "
            f"signal '{integrator_allocation.signal_id}', "
            f"got {integrator_allocation.channels}"
        )
        integration_unit_index = integrator_allocation.channels[0]

        nc = NodeCollector(
            base=f"/{self.serial}/qachannels/{measurement.channel}/readout/multistate/"
        )

        nc.add("enable", 1)
        nc.add("zsync/packed", 1)
        qudit_path = f"qudits/{integration_unit_index}"
        nc.add(f"{qudit_path}/numstates", num_states)
        nc.add(f"{qudit_path}/enable", 1, cache=False)
        nc.add(
            f"{qudit_path}/assignmentvec",
            calc_theoretical_assignment_vec(num_states - 1),
        )

        return nc

    def _configure_readout_mode_nodes(
        self,
        _dev_input: IO,
        _dev_output: IO,
        measurement: Measurement | None,
        device_uid: str,
        recipe_data: RecipeData,
    ) -> NodeCollector:
        _logger.debug("%s: Setting measurement mode to 'Readout'.", self.dev_repr)
        assert measurement is not None

        nc = NodeCollector(
            base=f"/{self.serial}/qachannels/{measurement.channel}/readout/"
        )

        nc.add("integration/length", measurement.length)
        nc.add("multistate/qudits/*/enable", 0, cache=False)

        for integrator_allocation in recipe_data.recipe.integrator_allocations:
            if (
                integrator_allocation.device_id != device_uid
                or integrator_allocation.awg != measurement.channel
            ):
                continue
            readout_nodes = self._configure_readout_mode_nodes_multi_state(
                integrator_allocation, measurement
            )
            nc.extend(readout_nodes)

        return nc

    def _configure_spectroscopy_mode_nodes(
        self, dev_input: IO, measurement: Measurement | None
    ) -> NodeCollector:
        _logger.debug("%s: Setting measurement mode to 'Spectroscopy'.", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add(
            f"qachannels/{measurement.channel}/spectroscopy/trigger/channel",
            32 + measurement.channel,
        )
        nc.add(
            f"qachannels/{measurement.channel}/spectroscopy/length", measurement.length
        )

        return nc

    async def collect_awg_before_upload_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for measurement in initialization.measurements:
            nc.add(
                f"qachannels/{measurement.channel}/mode",
                0 if is_spectroscopy(acquisition_type) else 1,
            )

            dev_input = next(
                (
                    inp
                    for inp in initialization.inputs
                    if inp.channel == measurement.channel
                ),
                None,
            )
            dev_output = next(
                (
                    output
                    for output in initialization.outputs
                    if output.channel == measurement.channel
                ),
                None,
            )
            if is_spectroscopy(acquisition_type):
                nc.extend(
                    self._configure_spectroscopy_mode_nodes(dev_input, measurement)
                )
            elif acquisition_type != AcquisitionType.RAW:
                nc.extend(
                    self._configure_readout_mode_nodes(
                        dev_input,
                        dev_output,
                        measurement,
                        initialization.device_uid,
                        recipe_data,
                    )
                )

        return await self.maybe_async(nc)

    async def collect_awg_after_upload_nodes(
        self, initialization: Initialization
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        inputs = initialization.inputs or []
        for dev_input in inputs:
            nc.add(f"qachannels/{dev_input.channel}/input/on", 1)
            if dev_input.range is not None:
                self._validate_range(dev_input, is_out=False)
                nc.add(f"qachannels/{dev_input.channel}/input/range", dev_input.range)

        for measurement in initialization.measurements:
            channel = 0
            if initialization.config.triggering_mode == TriggeringMode.DESKTOP_LEADER:
                # standalone QA oder QC
                channel = (
                    SOFTWARE_TRIGGER_CHANNEL
                    if self.options.is_qc
                    else INTERNAL_TRIGGER_CHANNEL
                )
            nc.add(
                f"qachannels/{measurement.channel}/generator/auxtriggers/0/channel",
                channel,
            )

        return await self.maybe_async(nc)

    async def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        _logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        nc = NodeCollector(base=f"/{self.serial}/")

        triggering_mode = initialization.config.triggering_mode

        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            pass
        elif triggering_mode == TriggeringMode.DESKTOP_LEADER:
            self._wait_for_awgs = False
            self._emit_trigger = True
            if self.options.is_qc:
                nc.add("system/internaltrigger/enable", 0)
                nc.add("system/internaltrigger/repetitions", 1)
        else:
            raise LabOneQControllerException(
                f"Unsupported triggering mode: {triggering_mode} for device type SHFQA."
            )

        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            src = 32 + awg_index
            nc.add(f"qachannels/{awg_index}/markers/0/source", src)
            nc.add(f"qachannels/{awg_index}/markers/1/source", src)
        return await self.maybe_async(nc)

    async def get_measurement_data(
        self,
        recipe_data: RecipeData,
        channel: int,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ):
        assert len(result_indices) == 1
        result_path = f"/{self.serial}/qachannels/{channel}/" + (
            "spectroscopy/result/data/wave"
            if is_spectroscopy(rt_execution_info.acquisition_type)
            else f"readout/result/data/{result_indices[0]}/wave"
        )
        ch_repr = (
            f"{self.dev_repr}:ch{channel}:spectroscopy"
            if is_spectroscopy(rt_execution_info.acquisition_type)
            else f"{self.dev_repr}:ch{channel}:readout{result_indices[0]}"
        )

        pipeliner_jobs = (
            rt_execution_info.pipeliner_jobs
            if recipe_data.setup_caps.result_logger_pipelined
            else 1
        )

        rt_result: npt.ArrayLike = np.empty(
            pipeliner_jobs * num_results, dtype=np.complex128
        )
        rt_result[:] = np.nan
        jobs_processed: set[int] = set()

        expected_job_id = 0  # TODO(2K): For compatibility with 23.10

        read_result_timeout_s = 5
        last_result_received = None
        while True:
            job_result = self.node_monitor.pop(result_path)
            if job_result is None:
                if len(jobs_processed) == pipeliner_jobs:
                    break
                now = time.monotonic()
                if last_result_received is None:
                    last_result_received = now
                if now - last_result_received > read_result_timeout_s:
                    _logger.error(
                        f"{ch_repr}: Failed to receive all results within {read_result_timeout_s} s, timing out."
                    )
                    break
                await asyncio.sleep(0.1)
                await self.node_monitor.poll()
                continue
            else:
                last_result_received = None

            job_id = job_result["properties"].get("jobid", expected_job_id)
            expected_job_id += 1
            if job_id in jobs_processed:
                _logger.error(
                    f"{ch_repr}: Ignoring duplicate job id {job_id} in the results."
                )
                continue
            if job_id >= pipeliner_jobs:
                _logger.error(
                    f"{ch_repr}: Ignoring job id {job_id} in the results, as it "
                    f"falls outside the defined range of {pipeliner_jobs} jobs."
                )
                continue
            jobs_processed.add(job_id)

            num_samples = job_result["properties"].get("numsamples", num_results)

            if num_samples != num_results:
                _logger.error(
                    f"{ch_repr}: The number of measurements acquired ({num_samples}) "
                    f"does not match the number of measurements defined ({num_results}). "
                    "Possibly the time between measurements within a loop is too short, "
                    "or the measurement was not started."
                )

            valid_samples = min(num_results, num_samples)
            np.put(
                rt_result,
                range(job_id * num_results, job_id * num_results + valid_samples),
                job_result["vector"][:valid_samples],
                mode="clip",
            )

        missing_jobs = set(range(pipeliner_jobs)) - jobs_processed
        if len(missing_jobs) > 0:
            _logger.error(
                f"{ch_repr}: Results for job id(s) {missing_jobs} are missing."
            )

        if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION:
            return rt_result.real
        return rt_result

    async def get_input_monitor_data(self, channel: int, num_results: int):
        result_path_ch = f"/{self.serial}/scopes/0/channels/{channel}/wave"
        node_data = await self.get_raw(result_path_ch)
        data = node_data[result_path_ch][0]["vector"][0:num_results]
        return data

    async def collect_reset_nodes(self) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        # Reset pipeliner first, attempt to set generator enable leads to FW error if pipeliner was enabled.
        nc.extend(self.pipeliner_reset_nodes())
        nc.add("qachannels/*/generator/enable", 0, cache=False)
        nc.add("system/synchronization/source", 0, cache=False)  # internal
        if self.options.is_qc:
            nc.add("system/internaltrigger/synchronization/enable", 0, cache=False)
        nc.add("qachannels/*/readout/result/enable", 0, cache=False)
        nc.add("qachannels/*/spectroscopy/psd/enable", 0, cache=False)
        nc.add("qachannels/*/spectroscopy/result/enable", 0, cache=False)
        nc.add("qachannels/*/output/rflfinterlock", 1, cache=False)
        nc.add("scopes/0/enable", 0, cache=False)
        nc.add("scopes/0/channels/*/enable", 0, cache=False)
        reset_nodes = await super().collect_reset_nodes()
        reset_nodes.extend(await self.maybe_async(nc))
        return reset_nodes
