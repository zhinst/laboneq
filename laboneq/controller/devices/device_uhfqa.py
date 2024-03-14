# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.communication import (
    DaqNodeSetAction,
)
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    DeviceZI,
    delay_to_rounded_samples,
    IntegrationWeights,
    IntegrationWeightItem,
)
from laboneq.controller.devices.zi_node_monitor import (
    Command,
    Response,
    Setting,
    NodeControlBase,
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
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IO, Initialization, IntegratorAllocation, TriggeringMode
from laboneq.data.scheduled_experiment import CompilerArtifact, ArtifactsCodegen

_logger = logging.getLogger(__name__)

SAMPLE_FREQUENCY_HZ = 1.8e9
DELAY_NODE_GRANULARITY_SAMPLES = 4
DELAY_NODE_MAX_SAMPLES = 1020

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1

MAX_AVERAGES_RESULT_LOGGER = 1 << 17
MAX_AVERAGES_SCOPE = 1 << 15


class DeviceUHFQA(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "UHFQA"
        self.dev_opts = ["AWG", "DIG", "QA"]
        self._channels = 2
        self._use_internal_clock = True

    def _get_num_awgs(self) -> int:
        return 1

    def _osc_group_by_channel(self, channel: int) -> int:
        return channel

    def _get_next_osc_index(
        self, osc_group: int, previously_allocated: int
    ) -> int | None:
        if previously_allocated >= 1:
            return None
        return previously_allocated

    async def disable_outputs(
        self, outputs: set[int], invert: bool
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                nc.add(f"sigouts/{ch}/on", 0, cache=False)
        return await self.maybe_async(nc)

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        for awg in range(self._get_num_awgs()):
            nodes.append(f"/{self.serial}/awgs/{awg}/enable")
            nodes.append(f"/{self.serial}/awgs/{awg}/ready")
        return nodes

    def _error_ambiguous_upstream(self):
        raise LabOneQControllerException(
            f"{self.dev_repr}: Can't determine unambiguously upstream device for UHFQA, ensure "
            f"correct DIO connection in the device setup"
        )

    def update_clock_source(self, force_internal: bool | None):
        if len(self._uplinks) == 0:
            raise LabOneQControllerException(
                f"{self.dev_repr}: UHFQA cannot be configured as leader, ensure correct DIO "
                f"connection in the device setup"
            )
        if len(self._uplinks) > 1:
            self._error_ambiguous_upstream()
        upstream = next(iter(self._uplinks))()
        if upstream is None:
            self._error_ambiguous_upstream()
        is_desktop = upstream.is_leader() and (
            upstream.device_qualifier.driver.upper() == "HDAWG"
        )
        # For non-desktop, always use external clock,
        # for desktop - internal is the default (force_internal is None),
        # but allow override to external.
        self._use_internal_clock = is_desktop and (force_internal is not False)

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        source = (
            REFERENCE_CLOCK_SOURCE_INTERNAL
            if self._use_internal_clock
            else REFERENCE_CLOCK_SOURCE_EXTERNAL
        )
        return [
            Setting(f"/{self.serial}/system/extclk", source),
        ]

    def load_factory_preset_control_nodes(self) -> list[NodeControlBase]:
        return [
            Command(f"/{self.serial}/system/preset/index", 0),
            Command(f"/{self.serial}/system/preset/load", 1),
            Response(f"/{self.serial}/system/preset/busy", 0),
        ]

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
        nc.extend(
            self._configure_result_logger(
                awg_key,
                awg_config,
                integrator_allocations,
                averages,
                averaging_mode,
                acquisition_type,
            )
        )
        nc.extend(
            self._configure_input_monitor(
                enable=acquisition_type == AcquisitionType.RAW,
                averages=averages,
                acquire_length=awg_config.raw_acquire_length,
            )
        )
        return await self.maybe_async(nc)

    def _configure_result_logger(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        enable = acquisition_type != AcquisitionType.RAW
        if enable:
            if averaging_mode != AveragingMode.SINGLE_SHOT:
                if averages > MAX_AVERAGES_RESULT_LOGGER:
                    raise LabOneQControllerException(
                        f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                    )
                if averages & (averages - 1):
                    raise LabOneQControllerException(
                        f"Number of averages {averages} must be a power of 2"
                    )
            nc.add("qas/0/result/length", awg_config.result_length)
            nc.add("qas/0/result/averages", averages)
            nc.add(
                "qas/0/result/mode", 0 if averaging_mode == AveragingMode.CYCLIC else 1
            )
            nc.add(
                "qas/0/result/source",
                # 1 == result source 'threshold'
                # 2 == result source 'rotation'
                1 if acquisition_type == AcquisitionType.DISCRIMINATION else 2,
            )
            nc.add("qas/0/result/enable", 0)
            nc.add("qas/0/result/reset", 1, cache=False)
        _logger.debug("Turning %s result logger...", "on" if enable else "off")
        nc.add("qas/0/result/enable", 1 if enable else 0)
        return nc

    def _configure_input_monitor(
        self, enable: bool, averages: int, acquire_length: int
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        if enable:
            if averages > MAX_AVERAGES_SCOPE:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_SCOPE}"
                )
            nc.add("qas/0/monitor/length", acquire_length)
            nc.add("qas/0/monitor/averages", averages)
            nc.add("qas/0/monitor/enable", 0)
            nc.add("qas/0/monitor/reset", 1, cache=False)
        nc.add("qas/0/monitor/enable", 1 if enable else 0)
        return nc

    async def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, Any]:
        conditions: dict[str, Any] = {}
        for awg_index in self._allocated_awgs:
            conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = 1
        return await self.maybe_async_wait(conditions)

    async def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, Any]:
        conditions: dict[str, Any] = {}
        for awg_index in self._allocated_awgs:
            conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = 0
        return await self.maybe_async_wait(conditions)

    def _validate_range(self, io: IO, is_out: bool):
        if io.range is None:
            return

        input_ranges = np.concatenate(
            [np.arange(0.01, 0.1, 0.01), np.arange(0, 1.6, 0.1)]
        )
        output_ranges = np.array([0.15, 1.5], dtype=np.float64)
        range_list = output_ranges if is_out else input_ranges
        label = "Output" if is_out else "Input"

        if io.range_unit not in (None, "volt"):
            raise LabOneQControllerException(
                f"{label} range of device {self.dev_repr} is specified in "
                f"units of {io.range_unit}. Units must be 'volt'."
            )

        if not any(np.isclose([io.range] * len(range_list), range_list)):
            _logger.warning(
                "%s: %s channel %d range %.1f is not on the list of allowed ranges: %s. Nearest "
                "allowed range will be used.",
                self.dev_repr,
                label,
                io.channel,
                io.range,
                range_list,
            )

    def _validate_initialization(self, initialization: Initialization):
        super()._validate_initialization(initialization)
        outputs = initialization.outputs or []
        for output in outputs:
            if output.port_delay is not None:
                if output.port_delay != 0:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}'s output does not support port delay"
                    )
                _logger.debug(
                    "%s's output port delay should be set to None, not 0", self.dev_repr
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
                output.gains is None, "correction_matrix", output.channel
            )

            awg_idx = output.channel // 2
            self._allocated_awgs.add(awg_idx)

            nc.add(f"sigouts/{output.channel}/on", 1 if output.enable else 0)
            if output.enable:
                nc.add(f"sigouts/{output.channel}/imp50", 1)
            nc.add(f"sigouts/{output.channel}/offset", output.offset)

            nc.add(f"awgs/{awg_idx}/single", 1)

            # the following is needed so that in spectroscopy mode, pulse lengths are correct
            # TODO(2K): Why 2 enables per sigout, but only one is used?
            nc.add(f"sigouts/{output.channel}/enables/{output.channel}", 1)

            nc.add(
                f"awgs/{awg_idx}/outputs/{output.channel}/mode",
                1 if output.modulation else 0,
            )

            if output.range is not None:
                self._validate_range(output, is_out=True)
                nc.add(f"sigouts/{output.channel}/range", output.range)

        return await self.maybe_async(nc)

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.extend(super().collect_prepare_nt_step_nodes(attributes, recipe_data))

        for ch in range(self._channels):
            [scheduler_port_delay, port_delay], updated = attributes.resolve(
                keys=[
                    (AttributeName.INPUT_SCHEDULER_PORT_DELAY, ch),
                    (AttributeName.INPUT_PORT_DELAY, ch),
                ]
            )
            if not updated or scheduler_port_delay is None:
                continue

            measurement_delay = scheduler_port_delay + (port_delay or 0.0)
            measurement_delay_rounded = delay_to_rounded_samples(
                channel=ch,
                dev_repr=self.dev_repr,
                delay=measurement_delay,
                sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                max_node_delay_samples=DELAY_NODE_MAX_SAMPLES,
            )

            nc.add("qas/0/delay", measurement_delay_rounded)

        return nc

    def _choose_wf_collector(
        self, elf_nodes: NodeCollector, wf_nodes: NodeCollector
    ) -> NodeCollector:
        return wf_nodes

    def _elf_upload_condition(self, awg_index: int) -> dict[str, Any]:
        # UHFQA does not yet support upload of ELF and waveforms in a single transaction.
        ready_node = self.get_sequencer_paths(awg_index).ready
        return {ready_node: 1}

    def _adjust_frequency(self, freq):
        # To make the phase correct on the UHFQA (q leading i channel by 90 degrees)
        # we need to flip the sign of the oscillator frequency
        return freq * -1.0

    def _configure_standard_mode_nodes(
        self,
        acquisition_type: AcquisitionType,
        device_uid: str,
        recipe_data: RecipeData,
    ) -> NodeCollector:
        _logger.debug("%s: Setting measurement mode to 'Standard'.", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")

        nc.add("qas/0/integration/mode", 0)
        for integrator_allocation in recipe_data.recipe.integrator_allocations:
            if integrator_allocation.device_id != device_uid:
                continue

            # TODO(2K): RAW was treated same as integration, as once was considered for use in
            # parallel, but actually this is not the case, and integration settings are not needed
            # for RAW.
            if acquisition_type in [AcquisitionType.INTEGRATION, AcquisitionType.RAW]:
                if len(integrator_allocation.channels) != 2:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Internal error - expected 2 integrators for signal "
                        f"'{integrator_allocation.signal_id}' in integration mode, "
                        f"got {len(integrator_allocation.channels)}"
                    )

                # 0: 1 -> Real, 2 -> Imag
                # 1: 2 -> Real, 1 -> Imag
                inputs_mapping = [0, 1]
                rotations = [1 + 1j, 1 - 1j]
            else:
                if len(integrator_allocation.channels) != 1:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}: Internal error - expected 1 integrator for signal "
                        f"'{integrator_allocation.signal_id}', "
                        f"got {len(integrator_allocation.channels)}"
                    )
                # 0: 1 -> Real, 2 -> Imag
                inputs_mapping = [0]
                rotations = [1 + 1j]

            for integrator, integration_unit_index in enumerate(
                integrator_allocation.channels
            ):
                nc.add(
                    f"qas/0/integration/sources/{integration_unit_index}",
                    inputs_mapping[integrator],
                )
                nc.add(
                    f"qas/0/rotations/{integration_unit_index}", rotations[integrator]
                )
                if acquisition_type in [
                    AcquisitionType.INTEGRATION,
                    AcquisitionType.DISCRIMINATION,
                ]:
                    nc.add(
                        f"qas/0/thresholds/{integration_unit_index}/correlation/enable",
                        0,
                    )
                    nc.add(
                        f"qas/0/thresholds/{integration_unit_index}/level",
                        integrator_allocation.thresholds[0] or 0.0,
                    )

        return nc

    def _configure_spectroscopy_mode_nodes(self) -> NodeCollector:
        _logger.debug("%s: Setting measurement mode to 'Spectroscopy'.", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("qas/0/integration/mode", 1)
        nc.add("qas/0/integration/sources/0", 1)
        nc.add("qas/0/integration/sources/1", 0)

        # The rotation coefficients in spectroscopy mode have to take into account that I and Q are
        # swapped between in- and outputs, i.e. the AWG outputs are I = AWG_wave_I * cos,
        # Q = AWG_wave_Q * sin, while the weights are I = sin and Q = cos. For more details,
        # see "Complex multiplication in UHFQA":
        # https://zhinst.atlassian.net/wiki/spaces/~andreac/pages/787742991/Complex+multiplication+in+UHFQA
        # https://oldwiki.zhinst.com/wiki/display/~andreac/Complex+multiplication+in+UHFQA)
        nc.add("qas/0/rotations/0", 1 - 1j)
        nc.add("qas/0/rotations/1", -1 - 1j)
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

        bin_waves: IntegrationWeights = []
        for signal_id, weight_names in integration_weights["signals"].items():
            integrator_allocation = next(
                ia for ia in integrator_allocations if ia.signal_id == signal_id
            )

            for index, weight_name in enumerate(weight_names):
                for channel in integrator_allocation.channels:
                    weight_vector_real = get_wave(
                        weight_name + "_i.wave", artifacts.waves
                    )
                    weight_vector_imag = get_wave(
                        weight_name + "_q.wave", artifacts.waves
                    )
                    bin_waves.append(
                        IntegrationWeightItem(
                            integration_unit=channel,
                            index=index,
                            name=weight_name,
                            # Note conjugation here:
                            samples=weight_vector_real - 1j * weight_vector_imag,
                        )
                    )

        return bin_waves

    def prepare_upload_all_integration_weights(
        self, awg_index, integration_weights: IntegrationWeights
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        for iw in integration_weights:
            nc.add(
                f"qas/0/integration/weights/{iw.integration_unit}/real",
                iw.samples.real,
                filename=iw.name + "_i.wave",
            )
            nc.add(
                f"qas/0/integration/weights/{iw.integration_unit}/imag",
                iw.samples.imag,
                filename=iw.name + "_q.wave",
            )
        return nc

    async def collect_awg_before_upload_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )
        if acquisition_type == AcquisitionType.SPECTROSCOPY_IQ:
            nc = self._configure_spectroscopy_mode_nodes()
        else:
            nc = self._configure_standard_mode_nodes(
                acquisition_type, initialization.device_uid, recipe_data
            )

        return await self.maybe_async(nc)

    async def collect_awg_after_upload_nodes(
        self, initialization: Initialization
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")
        inputs = initialization.inputs
        if len(initialization.measurements) > 0:
            measurement = initialization.measurements[0]

            _logger.debug(
                "%s: Setting measurement sample length to %d",
                self.dev_repr,
                measurement.length,
            )
            nc.add("qas/0/integration/length", measurement.length)

            nc.add("qas/0/integration/trigger/channel", 7)

        for dev_input in inputs or []:
            if dev_input.range is None:
                continue
            self._validate_range(dev_input, is_out=False)
            nc.add(f"sigins/{dev_input.channel}/range", dev_input.range)

        return await self.maybe_async(nc)

    async def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeSetAction]:
        nc = NodeCollector(base=f"/{self.serial}/")

        # Loop over at least AWG instance to cover the case that the instrument is only used as a
        # communication proxy. Some of the nodes on the AWG branch are needed to get proper
        # communication between HDAWG and UHFQA.
        for awg_index in (
            self._allocated_awgs if len(self._allocated_awgs) > 0 else range(1)
        ):
            nc.add(f"awgs/{awg_index}/dio/strobe/index", 16)
            nc.add(f"awgs/{awg_index}/dio/strobe/slope", 0)
            nc.add(f"awgs/{awg_index}/dio/valid/polarity", 2)
            nc.add(f"awgs/{awg_index}/dio/valid/index", 16)

        triggering_mode = initialization.config.triggering_mode

        if triggering_mode == TriggeringMode.DIO_FOLLOWER or triggering_mode is None:
            nc.add("dios/0/mode", 4)
            nc.add("dios/0/drive", 0x3)
            nc.add("dios/0/extclk", 0x2)
        elif triggering_mode == TriggeringMode.DESKTOP_DIO_FOLLOWER:
            nc.add("dios/0/mode", 0)
            nc.add("dios/0/drive", 0)
            nc.add("dios/0/extclk", 0x2)
            nc.add("awgs/0/auxtriggers/0/channel", 0)
            nc.add("awgs/0/auxtriggers/0/slope", 1)
        for trigger_index in (0, 1):
            nc.add(f"triggers/out/{trigger_index}/delay", 0.0)
            nc.add(f"triggers/out/{trigger_index}/drive", 1)
            nc.add(f"triggers/out/{trigger_index}/source", 32 + trigger_index)

        return await self.maybe_async(nc)

    async def _get_integrator_measurement_data(
        self, result_index, num_results, averages_divider: int
    ):
        result_path = f"/{self.serial}/qas/0/result/data/{result_index}/wave"
        # @TODO(andreyk): replace the raw daq reply parsing on site here and hide it inside
        # Communication class
        data_node_query = await self.get_raw(result_path)
        assert len(data_node_query[result_path][0]["vector"]) == num_results, (
            f"{self.dev_repr}: number of measurement points returned"
            " does not match length of recipe measurement_map"
        )
        return data_node_query[result_path][0]["vector"] / averages_divider

    async def get_measurement_data(
        self,
        recipe_data: RecipeData,
        channel: int,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ):
        averages_divider = (
            1
            if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION
            else hw_averages
        )
        assert len(result_indices) <= 2
        if len(result_indices) == 1:
            return await self._get_integrator_measurement_data(
                result_indices[0], num_results, averages_divider
            )
        else:
            in_phase = await self._get_integrator_measurement_data(
                result_indices[0], num_results, averages_divider
            )
            quadrature = await self._get_integrator_measurement_data(
                result_indices[1], num_results, averages_divider
            )
            return [complex(real, imag) for real, imag in zip(in_phase, quadrature)]

    async def get_input_monitor_data(self, channel: int, num_results: int):
        result_path_ch0 = f"/{self.serial}/qas/0/monitor/inputs/0/wave".lower()
        result_path_ch1 = f"/{self.serial}/qas/0/monitor/inputs/1/wave".lower()
        data = await self.get_raw(",".join([result_path_ch0, result_path_ch1]))
        # Truncate returned vectors to the expected length -> hotfix for GCE-681
        ch0 = data[result_path_ch0][0]["vector"][0:num_results]
        ch1 = data[result_path_ch1][0]["vector"][0:num_results]
        return [complex(real, imag) for real, imag in zip(ch0, ch1)]

    async def check_results_acquired_status(
        self, channel, acquisition_type: AcquisitionType, result_length, hw_averages
    ):
        results_acquired_path = f"/{self.serial}/qas/0/result/acquired"
        batch_get_results = await self.get_raw_values(results_acquired_path)
        if batch_get_results[results_acquired_path] != 0:
            raise LabOneQControllerException(
                f"The number of measurements executed for device {self.serial} does not match "
                f"the number of measurements defined. Probably the time between measurements or "
                f"within a loop is too short. Please contact Zurich Instruments."
            )
