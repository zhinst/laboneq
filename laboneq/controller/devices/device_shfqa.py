# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import typing as npt

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeGetAction,
    DaqNodeSetAction,
)
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.recipe_1_4_0 import (
    IO,
    Initialization,
    IntegratorAllocation,
    Measurement,
)
from laboneq.controller.recipe_enums import DIOConfigType
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
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1
REFERENCE_CLOCK_SOURCE_ZSYNC = 2

INTERNAL_TRIGGER_CHANNEL = 1024  # PQSC style triggering on the SHFSG/QC
SOFTWARE_TRIGGER_CHANNEL = 8  # Software triggering on the SHFQA

SAMPLE_FREQUENCY_HZ = 2.0e9
DELAY_NODE_GRANULARITY_SAMPLES = 4
DELAY_NODE_MAX_SAMPLES = 1e-6 * SAMPLE_FREQUENCY_HZ
# About DELAY_NODE_MAX_SAMPLES: The max time is actually 131e-6 s (at least I can set that
# value in GUI and API). However, there were concerns that these long times are not tested
# often enough - also, if you read the value back from the API, some lesser significant bits
# have strange values which looked a bit suspicious. Therefore, it was decided to limit the
# maximum delay to 1 us for now


class DeviceSHFQA(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "SHFQA4"
        self.dev_opts = []
        self._channels = 4
        self._wait_for_awgs = True
        self._emit_trigger = False

    @property
    def dev_repr(self) -> str:
        if self._get_option("is_qc"):
            return f"SHFQC/QA:{self._get_option('serial')}"
        return f"SHFQA:{self._get_option('serial')}"

    def _process_dev_opts(self):
        if self.dev_type == "SHFQA4":
            self._channels = 4
        elif self.dev_type == "SHFQA2":
            self._channels = 2
        elif self.dev_type == "SHFQC":
            self._channels = 1
        else:
            self._logger.warning(
                "%s: Unknown device type '%s', assuming SHFQA4 device.",
                self.dev_repr,
                self.dev_type,
            )
            self._channels = 4

    def _get_sequencer_type(self) -> str:
        return "qa"

    def _get_sequencer_path_patterns(self) -> dict:
        return {
            "elf": "/{serial}/qachannels/{index}/generator/elf/data",
            "progress": "/{serial}/qachannels/{index}/generator/elf/progress",
            "enable": "/{serial}/qachannels/{index}/generator/enable",
            "ready": "/{serial}/qachannels/{index}/generator/ready",
        }

    def _get_num_awgs(self):
        return self._channels

    def _validate_range(self, io: IO.Data, is_out: bool):
        if io.range is None:
            return
        input_ranges = np.array(
            [-50, -30, -25, -20, -15, -10, -5, 0, 5, 10], dtype=np.float64
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
            self._logger.warning(
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
    ) -> Optional[int]:
        if previously_allocated >= 1:
            return None
        return previously_allocated

    def _make_osc_path(self, channel: int, index: int) -> str:
        return f"/{self.serial}/qachannels/{channel}/oscs/{index}/freq"

    def _nodes_to_monitor_impl(self) -> List[str]:
        nodes = []
        for awg in range(self._get_num_awgs()):
            nodes.extend(
                [
                    f"/{self.serial}/qachannels/{awg}/generator/enable",
                    f"/{self.serial}/qachannels/{awg}/generator/ready",
                    f"/{self.serial}/qachannels/{awg}/spectroscopy/result/enable",
                    f"/{self.serial}/qachannels/{awg}/readout/result/enable",
                ]
            )
        return nodes

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: List[IntegratorAllocation.Data],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> List[DaqNodeAction]:

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
                acquisition_type == AcquisitionType.SPECTROSCOPY,
                awg_key.awg_index,
                awg_config.result_length,
                averages,
                average_mode,
            ),
            *self._configure_scope(
                acquisition_type == AcquisitionType.RAW,
                awg_key.awg_index,
                averages,
                awg_config.acquire_length,
            ),
        ]
        return nodes

    def _configure_readout(
        self,
        acquisition_type: AcquisitionType,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: List[IntegratorAllocation.Data],
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
            if acquisition_type == AcquisitionType.DISCRIMINATION:
                for integrator in integrator_allocations:
                    if (
                        integrator.device_id != awg_key.device_uid
                        or integrator.signal_id not in awg_config.signals
                    ):
                        continue
                    assert len(integrator.channels) == 1
                    integrator_idx = integrator.channels[0]
                    nodes_to_initialize_readout.append(
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/qachannels/{channel}/readout/discriminators/"
                            f"{integrator_idx}/threshold",
                            integrator.threshold,
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
        enable: bool,
        channel: int,
        result_length: int,
        averages: int,
        average_mode: int,
    ):
        nodes_to_initialize_spectroscopy = []
        if enable:
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
                        f"/{self.serial}/qachannels/{channel}/spectroscopy/result/enable",
                        0,
                    ),
                ]
            )
        nodes_to_initialize_spectroscopy.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{channel}/spectroscopy/result/enable",
                1 if enable else 0,
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
                    # TODO(2K): 200ns input-to-output delay was taken from one of the example
                    # notebooks, what value to use?
                    DaqNodeSetAction(
                        self._daq, f"/{self.serial}/scopes/0/trigger/delay", 200e-9
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

    def collect_execution_nodes(self):
        self._logger.debug("Starting execution...")
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{awg_index}/generator/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for awg_index in self._allocated_awgs
        ]

    def collect_start_execution_nodes(self):
        if self._emit_trigger:
            return [
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/internaltrigger/enable"
                    if self._get_option("is_qc")
                    else f"/{self.serial}/system/swtriggers/0/single",
                    1,
                    caching_strategy=CachingStrategy.NO_CACHE,
                )
            ]
        return []

    def conditions_for_execution_ready(self) -> Dict[str, Any]:
        # TODO(janl): Not sure whether we need this condition this on the SHFQA (including SHFQC)
        # as well. The state of the generator enable wasn't always pickup up reliably, so we
        # only check in cases where we rely on external triggering mechanisms.
        conditions: Dict[str, Any] = {}
        if self._wait_for_awgs:
            for awg_index in self._allocated_awgs:
                conditions[
                    f"/{self.serial}/qachannels/{awg_index}/generator/enable"
                ] = 1
        return conditions

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType
    ) -> Dict[str, Any]:
        conditions: Dict[str, Any] = {}
        for awg_index in self._allocated_awgs:
            conditions[f"/{self.serial}/qachannels/{awg_index}/generator/enable"] = 0
            if acquisition_type == AcquisitionType.SPECTROSCOPY:
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

    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeSetAction]:
        self._logger.debug("%s: Initializing device...", self.dev_repr)

        nodes_to_initialize_output: List[DaqNodeSetAction] = []

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

            measurement_delay_rounded = (
                self._get_total_rounded_delay_samples(
                    output,
                    SAMPLE_FREQUENCY_HZ,
                    DELAY_NODE_GRANULARITY_SAMPLES,
                    DELAY_NODE_MAX_SAMPLES,
                    0,
                )
                / SAMPLE_FREQUENCY_HZ
            )

            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{output.channel}/generator/delay",
                    measurement_delay_rounded,
                )
            )
            nodes_to_initialize_output.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{output.channel}/spectroscopy/envelope/delay",
                    measurement_delay_rounded,
                )
            )

        return nodes_to_initialize_output

    def prepare_upload_binary_wave(
        self,
        filename: str,
        waveform: npt.ArrayLike,
        awg_index: int,
        wave_index: int,
        acquisition_type: AcquisitionType,
    ):
        assert acquisition_type != AcquisitionType.SPECTROSCOPY or wave_index == 0
        return DaqNodeSetAction(
            self._daq,
            f"/{self.serial}/qachannels/{awg_index}/spectroscopy/envelope/wave"
            if acquisition_type == AcquisitionType.SPECTROSCOPY
            else f"/{self.serial}/qachannels/{awg_index}/generator/waveforms/{wave_index}/wave",
            waveform,
            filename=filename,
            caching_strategy=CachingStrategy.NO_CACHE,
        )

    def prepare_upload_all_binary_waves(
        self,
        awg_index,
        waves: List[Tuple[str, npt.ArrayLike]],
        acquisition_type: AcquisitionType,
    ):
        waves_upload: List[DaqNodeSetAction] = []
        has_spectroscopy_envelope = False
        if acquisition_type == AcquisitionType.SPECTROSCOPY:
            if len(waves) > 1:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Only one envelope waveform per channel is possible in "
                    f"spectroscopy mode. Check play commands for channel {awg_index}."
                )
            max_len = 65536
            for wave_index, (filename, waveform) in enumerate(waves):
                has_spectroscopy_envelope = True
                wave_len = len(waveform)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the envelope waveform "
                        f"'{filename}' for spectroscopy unit {awg_index} exceeds maximum "
                        f"of {max_len} samples. Ensure measure pulse doesn't "
                        f"exceed {max_pulse_len * 1e6:.3f} us."
                    )
                waves_upload.append(
                    self.prepare_upload_binary_wave(
                        filename=filename,
                        waveform=waveform,
                        awg_index=awg_index,
                        wave_index=wave_index,
                        acquisition_type=acquisition_type,
                    )
                )
        else:
            max_len = 4096
            for wave_index, (filename, waveform) in enumerate(waves):
                wave_len = len(waveform)
                if wave_len > max_len:
                    max_pulse_len = max_len / SAMPLE_FREQUENCY_HZ
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Length {wave_len} of the waveform '{filename}' "
                        f"for generator {awg_index} / wave slot {wave_index} exceeds maximum "
                        f"of {max_len} samples. Ensure measure pulse doesn't exceed "
                        f"{max_pulse_len * 1e6:.3f} us."
                    )
                waves_upload.append(
                    self.prepare_upload_binary_wave(
                        filename=filename,
                        waveform=waveform,
                        awg_index=awg_index,
                        wave_index=wave_index,
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

    def _configure_readout_mode_nodes(
        self,
        dev_input: IO.Data,
        dev_output: IO.Data,
        measurement: Optional[Measurement.Data],
        device_uid: str,
        recipe_data: RecipeData,
    ):
        self._logger.debug("%s: Setting measurement mode to 'Readout'.", self.dev_repr)

        measurement_delay_output = 0
        if dev_output is not None:
            if dev_output.port_delay is not None:
                measurement_delay_output += dev_output.port_delay * SAMPLE_FREQUENCY_HZ

        measurement_delay_rounded = (
            self._get_total_rounded_delay_samples(
                dev_input,
                SAMPLE_FREQUENCY_HZ,
                DELAY_NODE_GRANULARITY_SAMPLES,
                DELAY_NODE_MAX_SAMPLES,
                measurement.delay + measurement_delay_output,
            )
            / SAMPLE_FREQUENCY_HZ
        )

        nodes_to_set_for_readout_mode = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/readout/integration/length",
                measurement.length,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/readout/integration/delay",
                measurement_delay_rounded,
            ),
        ]

        max_len = 4096
        for (
            integrator_allocation
        ) in recipe_data.recipe.experiment.integrator_allocations:
            if (
                integrator_allocation.device_id != device_uid
                or integrator_allocation.awg != measurement.channel
            ):
                continue
            if integrator_allocation.weights is None:
                # Skip configuration if no integration weights provided to keep same behavior
                # TODO(2K): Consider not emitting the integrator allocation in this case.
                continue

            if len(integrator_allocation.channels) != 1:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Internal error - expected 1 integrator for "
                    f"signal '{integrator_allocation.signal_id}', "
                    f"got {len(integrator_allocation.channels)}"
                )
            integration_unit_index = integrator_allocation.channels[0]
            wave_name = integrator_allocation.weights + ".wave"
            weight_vector = np.conjugate(
                get_wave(wave_name, recipe_data.compiled.waves)
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
            node_path = (
                f"/{self.serial}/qachannels/{measurement.channel}/readout/integration/"
                f"weights/{integration_unit_index}/wave"
            )
            nodes_to_set_for_readout_mode.append(
                DaqNodeSetAction(
                    self._daq,
                    node_path,
                    weight_vector,
                    filename=wave_name,
                    caching_strategy=CachingStrategy.CACHE,
                )
            )
        return nodes_to_set_for_readout_mode

    def _configure_spectroscopy_mode_nodes(
        self, dev_input, measurement: Optional[Measurement.Data]
    ):
        self._logger.debug(
            "%s: Setting measurement mode to 'Spectroscopy'.", self.dev_repr
        )

        measurement_delay_rounded = (
            self._get_total_rounded_delay_samples(
                dev_input,
                SAMPLE_FREQUENCY_HZ,
                DELAY_NODE_GRANULARITY_SAMPLES,
                DELAY_NODE_MAX_SAMPLES,
                measurement.delay,
            )
            / SAMPLE_FREQUENCY_HZ
        )

        nodes_to_set_for_spectroscopy_mode = [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/spectroscopy/trigger/channel",
                32 + measurement.channel,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/spectroscopy/delay",
                measurement_delay_rounded,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/qachannels/{measurement.channel}/spectroscopy/length",
                measurement.length,
            ),
        ]

        return nodes_to_set_for_spectroscopy_mode

    def collect_awg_before_upload_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ):
        nodes_to_initialize_measurement = []

        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
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
                else:
                    center_frequencies[io.channel] = idx
                    nodes_to_initialize_measurement.append(
                        DaqNodeSetAction(
                            self._daq,
                            f"/{self.serial}/qachannels/{io.channel}/centerfreq",
                            io.lo_frequency,
                        )
                    )

        for measurement in initialization.measurements:
            nodes_to_initialize_measurement.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/qachannels/{measurement.channel}/mode",
                    0 if acquisition_type == AcquisitionType.SPECTROSCOPY else 1,
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
            if acquisition_type == AcquisitionType.SPECTROSCOPY:
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

    def collect_awg_after_upload_nodes(self, initialization: Initialization.Data):
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
            if initialization.config.dio_mode == DIOConfigType.HDAWG_LEADER:
                # standalone QA oder QC
                channel = (
                    SOFTWARE_TRIGGER_CHANNEL
                    if self._get_option("is_qc")
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

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> List[DaqNodeAction]:
        self._logger.debug("Configuring triggers...")
        self._wait_for_awgs = True
        self._emit_trigger = False

        nodes_to_configure_triggers = []

        dio_mode = initialization.config.dio_mode

        if dio_mode == DIOConfigType.ZSYNC_DIO:
            pass
        elif dio_mode == DIOConfigType.HDAWG_LEADER:
            self._wait_for_awgs = False
            self._emit_trigger = True
            clock_source = initialization.config.reference_clock_source
            ntc = [
                (
                    "system/clocks/referenceclock/in/source",
                    REFERENCE_CLOCK_SOURCE_INTERNAL
                    if clock_source
                    and clock_source.value == ReferenceClockSource.INTERNAL.value
                    else REFERENCE_CLOCK_SOURCE_EXTERNAL,
                )
            ]
            if self._get_option("is_qc"):
                ntc += [
                    ("system/internaltrigger/enable", 0),
                    ("system/internaltrigger/repetitions", 1),
                ]
            return [
                DaqNodeSetAction(self._daq, f"/{self.serial}/{node}", v)
                for node, v in ntc
            ]

        else:
            raise LabOneQControllerException(
                f"Unsupported DIO mode: {dio_mode} for device type SHFQA."
            )

        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            marker_path = f"/{self.serial}/qachannels/{awg_index}/markers"
            src = 32 + awg_index
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"{marker_path}/0/source", src),
            )
            nodes_to_configure_triggers.append(
                DaqNodeSetAction(self._daq, f"{marker_path}/1/source", src),
            )
        return nodes_to_configure_triggers

    def configure_as_leader(self, initialization: Initialization.Data):
        raise LabOneQControllerException("SHFQA cannot be configured as leader")

    def collect_follower_configuration_nodes(
        self, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        dio_mode = initialization.config.dio_mode
        self._logger.debug("%s: Configuring as a follower...", self.dev_repr)

        nodes_to_configure_as_follower = []

        if dio_mode == DIOConfigType.ZSYNC_DIO:
            self._logger.debug(
                "%s: Configuring reference clock to use ZSYNC as a reference...",
                self.dev_repr,
            )
            self._switch_reference_clock(source=2, expected_freqs=100e6)
        elif dio_mode == DIOConfigType.HDAWG_LEADER:
            # standalone
            pass
        else:
            raise LabOneQControllerException(
                f"Unsupported DIO mode: {dio_mode} for device type SHFQA."
            )

        return nodes_to_configure_as_follower

    def get_measurement_data(
        self,
        channel: int,
        acquisition_type: AcquisitionType,
        result_indices: List[int],
        num_results: int,
        hw_averages: int,
    ):
        assert len(result_indices) == 1
        result_path = f"/{self.serial}/qachannels/{channel}/" + (
            "spectroscopy/result/data/wave"
            if acquisition_type == AcquisitionType.SPECTROSCOPY
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

    def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        unit = (
            "spectroscopy"
            if acquisition_type == AcquisitionType.SPECTROSCOPY
            else "readout"
        )
        results_acquired_path = (
            f"/{self.serial}/qachannels/{channel}/{unit}/result/acquired"
        )
        batch_get_results = self._daq.batch_get(
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

    def collect_reset_nodes(self) -> List[DaqNodeAction]:
        reset_nodes = super().collect_reset_nodes()
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
                f"/{self.serial}/qachannels/*/readout/result/enable",
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
