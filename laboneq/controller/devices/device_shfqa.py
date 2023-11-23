# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
    CachingStrategy,
    DaqNodeAction,
    DaqNodeGetAction,
    DaqNodeSetAction,
)
from laboneq.controller.devices.awg_pipeliner import AwgPipeliner
from laboneq.controller.devices.device_shf_base import DeviceSHFBase
from laboneq.controller.devices.device_zi import (
    SequencerPaths,
    Waveforms,
    delay_to_rounded_samples,
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


def node_generator(daq, l: list):
    def append(path, value, filename=None, cache=True):
        l.append(
            DaqNodeSetAction(
                daq=daq,
                path=path,
                value=value,
                filename=filename,
                caching_strategy=(
                    CachingStrategy.CACHE if cache else CachingStrategy.NO_CACHE
                ),
            )
        )

    return append


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

    def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        channels_to_disable: list[DaqNodeSetAction] = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{ch}/output/on",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for ch in range(self._channels)
            if (ch in outputs) != invert
        ]
        return channels_to_disable

    def on_experiment_end(self):
        nodes = super().on_experiment_end()
        return [
            *nodes,
            # in CW spectroscopy mode, turn off the tone
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/*/spectroscopy/envelope/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
        ]

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        for awg in range(self._get_num_awgs()):
            nodes.extend(
                [
                    f"/{self.serial}/qachannels/{awg}/generator/enable",
                    f"/{self.serial}/qachannels/{awg}/generator/ready",
                    f"/{self.serial}/qachannels/{awg}/spectroscopy/psd/enable",
                    f"/{self.serial}/qachannels/{awg}/spectroscopy/result/enable",
                    f"/{self.serial}/qachannels/{awg}/readout/result/enable",
                ]
            )
            nodes.extend(self.pipeliner_control_nodes(awg))
        return nodes

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> list[DaqNodeAction]:
        average_mode = 0 if averaging_mode == AveragingMode.CYCLIC else 1
        nodes = [
            *self._configure_readout(
                acquisition_type,
                awg_key,
                awg_config,
                integrator_allocations,
                averages,
                average_mode,
            ),
            *self._configure_spectroscopy(
                acquisition_type,
                awg_key.awg_index,
                awg_config.result_length,
                averages,
                average_mode,
            ),
            *self._configure_scope(
                enable=acquisition_type == AcquisitionType.RAW,
                channel=awg_key.awg_index,
                averages=averages,
                acquire_length=awg_config.raw_acquire_length,
            ),
        ]
        return nodes

    def _configure_readout(
        self,
        acquisition_type: AcquisitionType,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        average_mode: int,
    ):
        enable = acquisition_type in [
            AcquisitionType.INTEGRATION,
            AcquisitionType.DISCRIMINATION,
        ]
        channel = awg_key.awg_index
        nodes_to_initialize_readout = []
        if enable:
            nodes_to_initialize_readout.extend(
                [
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/readout/result/length",
                        awg_config.result_length,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/readout/result/averages",
                        averages,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/readout/result/source",
                        # 1 - result_of_integration
                        # 3 - result_of_discrimination
                        3 if acquisition_type == AcquisitionType.DISCRIMINATION else 1,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/readout/result/mode",
                        average_mode,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/readout/result/enable",
                        0,
                    ),
                ]
            )
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
                        nodes_to_initialize_readout.append(
                            DaqNodeSetAction(
                                self._daq,
                                f"/{self.serial}/qachannels/{channel}/readout/multistate/qudits/"
                                f"{integrator_idx}/thresholds/{state_i}/value",
                                threshold or 0.0,
                            )
                        )
        nodes_to_initialize_readout.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{channel}/readout/result/enable",
                1 if enable else 0,
            )
        )
        return nodes_to_initialize_readout

    def _configure_spectroscopy(
        self,
        acq_type: AcquisitionType,
        channel: int,
        result_length: int,
        averages: int,
        average_mode: int,
    ):
        nodes_to_initialize_spectroscopy = []
        if is_spectroscopy(acq_type):
            nodes_to_initialize_spectroscopy.extend(
                [
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/spectroscopy/result/length",
                        result_length,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/spectroscopy/result/averages",
                        averages,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/spectroscopy/result/mode",
                        average_mode,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/spectroscopy/psd/enable",
                        0,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{channel}/spectroscopy/result/enable",
                        0,
                    ),
                ]
            )

        if acq_type == AcquisitionType.SPECTROSCOPY_PSD:
            nodes_to_initialize_spectroscopy.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{channel}/spectroscopy/psd/enable",
                    1,
                ),
            )

        nodes_to_initialize_spectroscopy.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{channel}/spectroscopy/result/enable",
                1 if is_spectroscopy(acq_type) else 0,
            )
        )
        return nodes_to_initialize_spectroscopy

    def _configure_scope(
        self, enable: bool, channel: int, averages: int, acquire_length: int
    ):
        # TODO(2K): multiple acquire events
        nodes_to_initialize_scope = []
        if enable:
            nodes_to_initialize_scope.extend(
                [
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/time", 0
                    ),  # 0 -> 2 GSa/s
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/averaging/enable", 1
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/averaging/count", averages
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/scopes/0/channels/{channel}/enable",
                        1,
                    ),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/scopes/0/channels/{channel}/inputselect",
                        channel,
                    ),  # channelN_signal_input
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/length", acquire_length
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/segments/enable", 0
                    ),
                    # TODO(2K): multiple acquire events per monitor
                    # DaqNodeSetAction(self._daq, f"/{self.serial}/scopes/0/segments/enable", 1),
                    # DaqNodeSetAction(self._daq, f"/{self.serial}/scopes/0/segments/count",
                    #                  measurement.result_length),
                    # TODO(2K): only one trigger is possible for all channels. Which one to use?
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/scopes/0/trigger/channel",
                        64 + channel,
                    ),  # channelN_sequencer_monitor0
                    # TODO(caglark): HBAR-1779
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/scopes/0/trigger/delay",
                        SCOPE_DELAY_OFFSET,
                    ),
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/trigger/enable", 1
                    ),
                    DaqNodeSetAction(self._daq, f"/{self.serial}/scopes/0/enable", 0),
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/scopes/0/single",
                        1,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    ),
                ]
            )
        nodes_to_initialize_scope.append(
            DaqNodeSetAction(
                self._daq, f"/{self.serial}/scopes/0/enable", 1 if enable else 0
            )
        )
        return nodes_to_initialize_scope

    def collect_execution_nodes(self, with_pipeliner: bool) -> list[DaqNodeAction]:
        if with_pipeliner:
            return self.pipeliner_collect_execution_nodes()

        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{awg_index}/generator/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for awg_index in self._allocated_awgs
        ]

    def collect_execution_setup_nodes(
        self, with_pipeliner: bool, has_awg_in_use: bool
    ) -> list[DaqNodeAction]:
        hw_sync = with_pipeliner and has_awg_in_use
        nodes = []
        if hw_sync and self._emit_trigger:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/synchronization/enable",
                    1,  # enable
                )
            )
        if hw_sync and not self._emit_trigger:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/synchronization/source",
                    1,  # external
                )
            )
        return nodes

    def collect_internal_start_execution_nodes(self):
        if self._emit_trigger:
            return [
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/enable"
                    if self.options.is_qc
                    else f"/{self.serial}/system/swtriggers/0/single",
                    1,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            ]
        return []

    def conditions_for_execution_ready(self, with_pipeliner: bool) -> dict[str, Any]:
        if with_pipeliner:
            return self.pipeliner_conditions_for_execution_ready()

        # TODO(janl): Not sure whether we need this condition on the SHFQA (including SHFQC)
        # as well. The state of the generator enable wasn't always picked up reliably, so we
        # only check in cases where we rely on external triggering mechanisms.
        return {
            f"/{self.serial}/qachannels/{awg_index}/generator/enable": 1
            for awg_index in self._allocated_awgs
        }

    def conditions_for_execution_done(
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

        for awg_index in self._allocated_awgs:
            if is_spectroscopy(acquisition_type):
                conditions[
                    f"/{self.serial}/qachannels/{awg_index}/spectroscopy/result/enable"
                ] = 0
            elif acquisition_type in [
                AcquisitionType.INTEGRATION,
                AcquisitionType.DISCRIMINATION,
            ]:
                conditions[
                    f"/{self.serial}/qachannels/{awg_index}/readout/result/enable"
                ] = 0
        return conditions

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

    def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ) -> list[DaqNodeSetAction]:
        _logger.debug("%s: Initializing device...", self.dev_repr)

        nodes_to_initialize_output: list[DaqNodeSetAction] = []

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
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{output.channel}/output/on",
                    1 if output.enable else 0,
                )
            )
            if output.range is not None:
                self._validate_range(output, is_out=True)
                nodes_to_initialize_output.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{output.channel}/output/range",
                        output.range,
                    )
                )

            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{output.channel}/generator/single",
                    1,
                )
            )

        nodes_to_initialize_output += [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{input.channel}/input/rflfpath",
                1  # RF
                if input.port_mode is None or input.port_mode == PortMode.RF.value
                else 0,  # LF
            )
            for input in initialization.inputs or []
        ]

        return nodes_to_initialize_output

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        nodes_to_set = super().collect_prepare_nt_step_nodes(attributes, recipe_data)

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for ch in range(self._channels):
            [synth_cf], synth_cf_updated = attributes.resolve(
                keys=[(AttributeName.QA_CENTER_FREQ, ch)]
            )
            if synth_cf_updated:
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{ch}/centerfreq",
                        synth_cf,
                    )
                )

            [out_amp], out_amp_updated = attributes.resolve(
                keys=[(AttributeName.QA_OUT_AMPLITUDE, ch)]
            )
            if out_amp_updated:
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{ch}/oscs/0/gain",
                        out_amp,
                    )
                )

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

            base_channel_path = f"/{self.serial}/qachannels/{ch}"
            if is_spectroscopy(acquisition_type):
                output_delay_path = f"{base_channel_path}/spectroscopy/envelope/delay"
                meas_delay_path = f"{base_channel_path}/spectroscopy/delay"
                measurement_delay += SPECTROSCOPY_DELAY_OFFSET
                max_generator_delay = DELAY_NODE_SPECTROSCOPY_ENVELOPE_MAX_SAMPLES
                max_integrator_delay = DELAY_NODE_SPECTROSCOPY_MAX_SAMPLES
            else:
                output_delay_path = f"{base_channel_path}/generator/delay"
                meas_delay_path = f"{base_channel_path}/readout/integration/delay"
                measurement_delay += output_delay
                measurement_delay += INTEGRATION_DELAY_OFFSET
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
                nodes_to_set.append(
                    DaqNodeSetAction(self._daq, output_delay_path, output_delay_rounded)
                )

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
                nodes_to_set.append(
                    DaqNodeSetAction(
                        self._daq, meas_delay_path, measurement_delay_rounded
                    )
                )

        return nodes_to_set

    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: npt.ArrayLike,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ):
        assert not is_spectroscopy(acquisition_type) or wave_index == 0
        return DaqNodeSetAction(
            self._daq,
            f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/wave"
            if is_spectroscopy(acquisition_type)
            else f"/{self.serial}/qachannels/{awg_index}/generator/waveforms/{wave_index}/wave",
            waveform,
            filename=filename,
            caching_strategy=CachingStrategy.NO_CACHE,
        )

    def prepare_upload_all_binary_waves(
        self,
        awg_index,
        waves: Waveforms,
        acquisition_type: AcquisitionType,
    ):
        waves_upload: list[DaqNodeSetAction] = []
        has_spectroscopy_envelope = False
        if is_spectroscopy(acquisition_type):
            if len(waves) > 1:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Only one envelope waveform per physical channel is "
                    f"possible in spectroscopy mode. Check play commands for channel {awg_index}."
                )
            max_len = 65536
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
                waves_upload.append(
                    self.prepare_upload_binary_wave(
                        filename=wave.name,
                        waveform=wave.samples,
                        awg_index=awg_index,
                        wave_index=0,
                        acquisition_type=acquisition_type,
                    )
                )
        else:
            max_len = 4096
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
                waves_upload.append(
                    self.prepare_upload_binary_wave(
                        filename=wave.name,
                        waveform=wave.samples,
                        awg_index=awg_index,
                        wave_index=wave.index,
                        acquisition_type=acquisition_type,
                    )
                )
        waves_upload.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/enable",
                1 if has_spectroscopy_envelope else 0,
            )
        )
        return waves_upload

    def _integrator_has_consistent_msd_num_state(
        self, integrator_allocation: IntegratorAllocation.Data
    ):
        num_states = len(integrator_allocation.weights) + 1
        num_thresholds = len(integrator_allocation.thresholds)
        num_expected_thresholds = (num_states - 1) * (num_states) / 2
        if num_thresholds != num_expected_thresholds:
            raise LabOneQControllerException(
                f"Multi discrimination configuration of experiment is not consistent. "
                f"Received num weights={len(integrator_allocation.weights)}, num thresholds={len(integrator_allocation.thresholds)}, "
                f"where num_weights should be n-1 and num_thresholds should be (n-1)*n/2 with n the number of states."
            )
        return True

    def _configure_readout_mode_nodes_multi_state(
        self,
        integrator_allocation: IntegratorAllocation.Data,
        recipe_data: RecipeData,
        measurement: Measurement.Data,
        max_len: int,
    ):
        ret_nodes: list[DaqNodeSetAction] = []
        num_states = len(integrator_allocation.weights) + 1
        assert self._integrator_has_consistent_msd_num_state(integrator_allocation)

        assert len(integrator_allocation.channels) == 1, (
            f"{self.dev_repr}: Internal error - expected 1 integrator for "
            f"signal '{integrator_allocation.signal_id}', "
            f"got {integrator_allocation.channels}"
        )
        integration_unit_index = integrator_allocation.channels[0]

        # Note: copying this from grimsel_multistate_demo jupyter notebook
        base_path = (
            f"/{self.serial}/qachannels/{measurement.channel}/readout/multistate"
        )
        qudit_path = f"{base_path}/qudits/{integration_unit_index}"

        node = node_generator(self._daq, ret_nodes)
        node(f"{base_path}/enable", 1)
        node(f"{base_path}/zsync/packed", 1)
        node(f"{qudit_path}/numstates", num_states)
        node(f"{qudit_path}/enable", 1, cache=False)
        node(
            f"{qudit_path}/assignmentvec",
            calc_theoretical_assignment_vec(num_states - 1),
        )

        for state_i in range(0, num_states - 1):
            wave_name = integrator_allocation.weights[state_i] + ".wave"
            weight_vector = np.conjugate(
                get_wave(wave_name, recipe_data.scheduled_experiment.waves)
            )
            wave_len = len(weight_vector)
            if wave_len > max_len:
                max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Length {wave_len} of the integration weight "
                    f"'{integration_unit_index}' of channel {measurement.channel} exceeds "
                    f"maximum of {max_len} samples. Ensure length of acquire kernels don't "
                    f"exceed {max_pulse_len * 1e6:.3f} us."
                )
            node(f"{qudit_path}/weights/{state_i}/wave", weight_vector, wave_name)

        return ret_nodes

    def _configure_readout_mode_nodes(
        self,
        _dev_input: IO,
        _dev_output: IO,
        measurement: Measurement | None,
        device_uid: str,
        recipe_data: RecipeData,
    ):
        _logger.debug("%s: Setting measurement mode to 'Readout'.", self.dev_repr)
        assert measurement is not None

        nodes_to_set_for_readout_mode: list[DaqNodeSetAction] = []

        base_path = f"/{self.serial}/qachannels/{measurement.channel}/readout"
        node = node_generator(self._daq, nodes_to_set_for_readout_mode)
        node(f"{base_path}/integration/length", measurement.length)
        node(f"{base_path}/multistate/qudits/*/enable", 0, cache=False)

        max_len = 4096
        for integrator_allocation in recipe_data.recipe.integrator_allocations:
            if (
                integrator_allocation.device_id != device_uid
                or integrator_allocation.awg != measurement.channel
            ):
                continue
            if integrator_allocation.weights == [None]:
                # Skip configuration if no integration weights provided to keep same behavior
                # TODO(2K): Consider not emitting the integrator allocation in this case.
                continue
            readout_nodes = self._configure_readout_mode_nodes_multi_state(
                integrator_allocation, recipe_data, measurement, max_len
            )
            nodes_to_set_for_readout_mode.extend(readout_nodes)

        return nodes_to_set_for_readout_mode

    def _configure_spectroscopy_mode_nodes(
        self, dev_input: IO, measurement: Measurement | None
    ):
        _logger.debug("%s: Setting measurement mode to 'Spectroscopy'.", self.dev_repr)

        nodes_to_set_for_spectroscopy_mode = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/spectroscopy/trigger/channel",
                32 + measurement.channel,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/spectroscopy/length",
                measurement.length,
            ),
        ]

        return nodes_to_set_for_spectroscopy_mode

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ):
        nodes_to_initialize_measurement = []

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )

        for measurement in initialization.measurements:
            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{measurement.channel}/mode",
                    0 if is_spectroscopy(acquisition_type) else 1,
                )
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
                nodes_to_initialize_measurement.extend(
                    self._configure_spectroscopy_mode_nodes(dev_input, measurement)
                )
            else:
                nodes_to_initialize_measurement.extend(
                    self._configure_readout_mode_nodes(
                        dev_input,
                        dev_output,
                        measurement,
                        initialization.device_uid,
                        recipe_data,
                    )
                )
        return nodes_to_initialize_measurement

    def collect_awg_after_upload_nodes(self, initialization: Initialization):
        nodes_to_initialize_measurement = []
        inputs = initialization.inputs or []
        for dev_input in inputs:
            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{dev_input.channel}/input/on",
                    1,
                )
            )
            if dev_input.range is not None:
                self._validate_range(dev_input, is_out=False)
                nodes_to_initialize_measurement.append(
                    DaqNodeSetAction(
                        self._daq,
                        f"/{self.serial}/qachannels/{dev_input.channel}/input/range",
                        dev_input.range,
                    )
                )

        for measurement in initialization.measurements:
            channel = 0
            if initialization.config.triggering_mode == TriggeringMode.DESKTOP_LEADER:
                # standalone QA oder QC
                channel = (
                    SOFTWARE_TRIGGER_CHANNEL
                    if self.options.is_qc
                    else INTERNAL_TRIGGER_CHANNEL
                )
            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{measurement.channel}/generator/"
                    f"auxtriggers/0/channel",
                    channel,
                )
            )

        return nodes_to_initialize_measurement

    async def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        _logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        nodes_to_configure_triggers = []

        triggering_mode = initialization.config.triggering_mode

        if triggering_mode == TriggeringMode.ZSYNC_FOLLOWER:
            pass
        elif triggering_mode == TriggeringMode.DESKTOP_LEADER:
            self._wait_for_awgs = False
            self._emit_trigger = True
            if self.options.is_qc:
                int_trig_base = f"/{self.serial}/system/internaltrigger"
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(self._daq, f"{int_trig_base}/enable", 0)
                )
                nodes_to_configure_triggers.append(
                    DaqNodeSetAction(self._daq, f"{int_trig_base}/repetitions", 1)
                )
        else:
            raise LabOneQControllerException(
                f"Unsupported triggering mode: {triggering_mode} for device type SHFQA."
            )

        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            markers_base = f"/{self.serial}/qachannels/{awg_index}/markers"
            src = 32 + awg_index
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"{markers_base}/0/source", src),
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"{markers_base}/1/source", src),
            )
        return nodes_to_configure_triggers

    def get_measurement_data(
        self,
        channel: int,
        acquisition_type: AcquisitionType,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ):
        assert len(result_indices) == 1
        result_path = f"/{self.serial}/qachannels/{channel}/" + (
            "spectroscopy/result/data/wave"
            if is_spectroscopy(acquisition_type)
            else f"readout/result/data/{result_indices[0]}/wave"
        )
        attempts = 3  # Hotfix HBAR-949
        while attempts > 0:
            attempts -= 1
            # @TODO(andreyk): replace the raw daq reply parsing on site here and hide it
            # inside Communication class
            data_node_query = self._daq.get_raw(result_path)
            actual_num_measurement_points = len(
                data_node_query[result_path][0]["vector"]
            )
            if actual_num_measurement_points < num_results:
                time.sleep(0.1)
                continue
            break
        assert actual_num_measurement_points == num_results, (
            f"number of measurement points {actual_num_measurement_points} returned by daq "
            f"from device '{self.dev_repr}' does not match length of recipe "
            f"measurement_map which is {num_results}"
        )
        result: npt.ArrayLike = data_node_query[result_path][0]["vector"]
        if acquisition_type == AcquisitionType.DISCRIMINATION:
            return result.real
        return result

    def get_input_monitor_data(self, channel: int, num_results: int):
        result_path_ch = f"/{self.serial}/scopes/0/channels/{channel}/wave"
        node_data = self._daq.get_raw(result_path_ch)
        data = node_data[result_path_ch][0]["vector"][0:num_results]
        return data

    async def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        unit = "spectroscopy" if is_spectroscopy(acquisition_type) else "readout"
        results_acquired_path = (
            f"/{self.serial}/qachannels/{channel}/{unit}/result/acquired"
        )
        batch_get_results = await self._daq.batch_get(
            [
                DaqNodeGetAction(
                    self._daq,
                    results_acquired_path,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            ]
        )
        actual_results = batch_get_results[results_acquired_path]
        expected_results = result_length * hw_averages
        if actual_results != expected_results:
            raise LabOneQControllerException(
                f"The number of measurements ({actual_results}) executed for device {self.serial} "
                f"on channel {channel} does not match the number of measurements "
                f"defined ({expected_results}). Probably the time between measurements or within "
                f"a loop is too short. Please contact Zurich Instruments."
            )

    def collect_reset_nodes(self) -> list[DaqNodeAction]:
        reset_nodes = super().collect_reset_nodes()
        # Reset pipeliner first, attempt to set generator enable leads to FW error if pipeliner was enabled.
        reset_nodes.extend(self.pipeliner_reset_nodes())
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/*/generator/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/system/synchronization/source",
                0,  # internal
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        if self.options.is_qc:
            reset_nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/synchronization/enable",
                    0,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/*/readout/result/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/*/spectroscopy/psd/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/*/spectroscopy/result/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/*/output/rflfinterlock",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/scopes/0/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        reset_nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/scopes/0/channels/*/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        return reset_nodes
