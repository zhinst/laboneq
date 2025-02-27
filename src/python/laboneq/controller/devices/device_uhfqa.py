# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import _gather, canonical_vector
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    AllocatedOscillator,
    DeviceBase,
    delay_to_rounded_samples,
    RawReadoutData,
)
from laboneq.controller.devices.node_control import (
    Command,
    Response,
    Setting,
    NodeControlBase,
)
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    RecipeData,
    RtExecutionInfo,
    get_initialization_by_device_uid,
    get_wave,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import (
    IO,
    IntegratorAllocation,
    OscillatorParam,
    TriggeringMode,
)
from laboneq.data.scheduled_experiment import ArtifactsCodegen, ScheduledExperiment

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray


_logger = logging.getLogger(__name__)

SAMPLE_FREQUENCY_HZ = 1.8e9
DELAY_NODE_GRANULARITY_SAMPLES = 4
DELAY_NODE_MAX_SAMPLES = 1020

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1

MAX_AVERAGES_RESULT_LOGGER = 1 << 17
MAX_AVERAGES_SCOPE = 1 << 15


class DeviceUHFQA(DeviceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "UHFQA"
        self.dev_opts = ["AWG", "DIG", "QA"]
        self._channels = 2
        self._integrators = 10
        self._use_internal_clock = True

    def _get_num_awgs(self) -> int:
        return 1

    def validate_scheduled_experiment(
        self, device_uid: str, scheduled_experiment: ScheduledExperiment
    ):
        initialization = get_initialization_by_device_uid(
            scheduled_experiment.recipe, device_uid
        )
        if initialization is not None:
            for output in initialization.outputs:
                if output.port_delay is not None:
                    if output.port_delay != 0:
                        raise LabOneQControllerException(
                            f"{self.dev_repr}'s output does not support port delay"
                        )
                    _logger.debug(
                        "%s's output port delay should be set to None, not 0",
                        self.dev_repr,
                    )

    def _get_next_osc_index(
        self,
        osc_group_oscs: list[AllocatedOscillator],
        osc_param: OscillatorParam,
        recipe_data: RecipeData,
    ) -> int | None:
        previously_allocated = len(osc_group_oscs)
        if previously_allocated >= 1:
            return None
        return previously_allocated

    async def disable_outputs(self, outputs: set[int], invert: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        for ch in range(self._channels):
            if (ch in outputs) != invert:
                nc.add(f"sigouts/{ch}/on", 0, cache=False)
        await self.set_async(nc)

    def _result_node_integrator(self, result_index: int):
        return f"/{self.serial}/qas/0/result/data/{result_index}/wave"

    def _result_node_monitor(self, ch: int):
        return f"/{self.serial}/qas/0/monitor/inputs/{ch}/wave"

    def _error_ambiguous_upstream(self) -> NoReturn:
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

    def configure_acquisition(
        self,
        awg_key: AwgKey,
        awg_config: AwgConfig,
        integrator_allocations: list[IntegratorAllocation],
        averages: int,
        averaging_mode: AveragingMode,
        acquisition_type: AcquisitionType,
        pipeliner_job: int | None,
        recipe_data: RecipeData,
    ) -> NodeCollector:
        nc = NodeCollector()
        assert pipeliner_job is None, "Pipelining not supported on UHFQA"
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
        return nc

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
        if awg_config.result_length is None:
            return nc  # this instrument is unused for acquiring results
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
        nc.barrier()
        nc.add("qas/0/result/enable", 1 if enable else 0)
        nc.barrier()
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
            nc.add("qas/0/monitor/enable", 0)  # todo: barrier needed?
            nc.add("qas/0/monitor/reset", 1, cache=False)
        nc.add("qas/0/monitor/enable", 1 if enable else 0)
        return nc

    async def start_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        for awg_index in self._allocated_awgs:
            nc.add(f"awgs/{awg_index}/enable", 1, cache=False)
        await self.set_async(nc)

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in self._allocated_awgs:
            conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                1,
                f"{self.dev_repr}: AWG {awg_index + 1} didn't start.",
            )
        return conditions

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in self._allocated_awgs:
            conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                0,
                f"{self.dev_repr}: AWG {awg_index + 1} didn't stop. Missing start trigger? Check DIO.",
            )
        return conditions

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

    async def apply_initialization(self, recipe_data: RecipeData):
        _logger.debug("%s: Initializing device...", self.dev_repr)

        nc = NodeCollector(base=f"/{self.serial}/")

        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
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

        await self.set_async(nc)

    def _collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.extend(super()._collect_prepare_nt_step_nodes(attributes, recipe_data))

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

    def prepare_upload_all_integration_weights(
        self,
        recipe_data: RecipeData,
        device_uid: str,
        awg_index: int,
        artifacts: ArtifactsCodegen,
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self.serial}/")

        integration_weights = artifacts.integration_weights.get(kernel_ref, {})
        for signal_id, weight_names in integration_weights.items():
            integrator_allocation = next(
                ia for ia in integrator_allocations if ia.signal_id == signal_id
            )

            for weight in weight_names:
                for channel in integrator_allocation.channels:
                    weight_wave_real = get_wave(weight.id + "_i.wave", artifacts.waves)
                    weight_wave_imag = get_wave(weight.id + "_q.wave", artifacts.waves)
                    nc.add(
                        f"qas/0/integration/weights/{channel}/real",
                        np.ascontiguousarray(weight_wave_real.samples),
                        filename=weight.id + "_i.wave",
                    )
                    nc.add(
                        f"qas/0/integration/weights/{channel}/imag",
                        # Note conjugation here
                        -np.ascontiguousarray(weight_wave_imag.samples),
                        filename=weight.id + "_q.wave",
                    )

        return nc

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        acquisition_type = RtExecutionInfo.get_acquisition_type(
            recipe_data.rt_execution_infos
        )
        if acquisition_type == AcquisitionType.SPECTROSCOPY_IQ:
            nc = self._configure_spectroscopy_mode_nodes()
        else:
            nc = self._configure_standard_mode_nodes(
                acquisition_type, self.device_qualifier.uid, recipe_data
            )
        await self.set_async(nc)

    async def set_after_awg_upload(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
        inputs = initialization.inputs
        if len(initialization.measurements) > 0:
            [measurement] = initialization.measurements

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

        await self.set_async(nc)

    async def configure_trigger(self, recipe_data: RecipeData):
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

        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
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

        await self.set_async(nc)

    async def on_experiment_begin(self):
        nodes = [
            *(
                self._result_node_integrator(result_index)
                for result_index in range(self._integrators)
            ),
            self._result_node_monitor(0),
            self._result_node_monitor(1),
        ]
        await _gather(*(self._subscriber.subscribe(self._api, node) for node in nodes))
        await super().on_experiment_begin()

    def _ch_repr_readout(self, ch: int) -> str:
        return f"{self.dev_repr}:readout{ch}"

    def _ch_repr_monitor(self, ch: int) -> str:
        return f"{self.dev_repr}:monitor{ch}"

    def _check_result(self, node_val: NumPyArray, num_results: int, ch_repr: str):
        num_samples = len(node_val)
        if num_samples != num_results:
            _logger.error(
                f"{ch_repr}: The number of measurements acquired ({num_samples}) "
                f"does not match the number of measurements defined ({num_results}). "
                "Possibly the time between measurements within a loop is too short, "
                "or the measurement was not started."
            )

    async def _get_integrator_measurement_data(
        self, result_index, num_results, averages_divider: int
    ):
        result_path = self._result_node_integrator(result_index)
        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5.0
        try:
            node_data = await self._subscriber.get(result_path, timeout_s=timeout_s)
            node_val = canonical_vector(node_data.value)
            self._check_result(
                node_val=node_val,
                num_results=num_results,
                ch_repr=self._ch_repr_readout(result_index),
            )
            # Not dividing by averages_divider - it appears poll data is already divided.
            return node_val[0:num_results]
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{self._ch_repr_readout(result_index)}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            return np.array([], dtype=np.complex128)

    async def get_measurement_data(
        self,
        recipe_data: RecipeData,
        channel: int,
        rt_execution_info: RtExecutionInfo,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ) -> RawReadoutData:
        averages_divider = (
            1
            if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION
            else hw_averages
        )
        assert len(result_indices) <= 2
        if len(result_indices) == 1:
            data = await self._get_integrator_measurement_data(
                result_indices[0], num_results, averages_divider
            )
            return RawReadoutData(data)
        else:
            in_phase, quadrature = await _gather(
                self._get_integrator_measurement_data(
                    result_indices[0], num_results, averages_divider
                ),
                self._get_integrator_measurement_data(
                    result_indices[1], num_results, averages_divider
                ),
            )
            return RawReadoutData(
                np.array(
                    [complex(real, imag) for real, imag in zip(in_phase, quadrature)]
                )
            )

    async def _get_input_monitor_data(self, ch: int, acquire_length: int):
        result_path = self._result_node_monitor(ch)
        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5.0
        try:
            node_data = await self._subscriber.get(result_path, timeout_s=timeout_s)
            node_val = canonical_vector(node_data.value)
            self._check_result(
                node_val=node_val,
                num_results=acquire_length,
                ch_repr=self._ch_repr_monitor(ch),
            )
            # Truncate returned vectors to the expected length -> hotfix for GCE-681
            return node_val[0:acquire_length]
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{self._ch_repr_monitor(ch)}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            return np.array([], dtype=np.complex128)

    async def get_raw_data(
        self, channel: int, acquire_length: int, acquires: int | None
    ) -> RawReadoutData:
        ch0, ch1 = await _gather(
            self._get_input_monitor_data(0, acquire_length),
            self._get_input_monitor_data(1, acquire_length),
        )
        return RawReadoutData(
            np.array([[complex(real, imag) for real, imag in zip(ch0, ch1)]])
        )
