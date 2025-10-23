# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Iterator
import logging
from typing import TYPE_CHECKING, Any, NoReturn

from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.uhfqa_awg import UHFQAAwg
from laboneq.controller.utilities.for_each import for_each
import numpy as np

from laboneq.controller.attribute_value_tracker import (
    AttributeName,
    DeviceAttributesView,
)
from laboneq.controller.devices.async_support import _gather
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
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
    RecipeData,
    RtExecutionInfo,
    get_execution_time,
    get_initialization_by_device_uid,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IO
from laboneq.data.scheduled_experiment import ScheduledExperiment

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
        self._awg_cores: list[UHFQAAwg] = []
        self._channels = 2
        self._integrators = 10
        self._use_internal_clock = True

    def all_channels(self) -> Iterator[ChannelBase]:
        """Iterable over all awg cores of the device."""
        return iter(self._awg_cores)

    def allocated_channels(self, recipe_data: RecipeData) -> Iterator[ChannelBase]:
        for ch in recipe_data.allocated_awgs(self.uid):
            yield self._awg_cores[ch]

    def _process_dev_opts(self):
        self._awg_cores = [
            UHFQAAwg(
                self._api,
                self._subscriber,
                self.uid,
                self.serial,
                repr_base=self.dev_repr,
            )
        ]

    def is_desktop(self) -> bool:
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
        return upstream.is_leader() and (
            upstream.device_qualifier.driver.upper() == "HDAWG"
        )

    def validate_scheduled_experiment(
        self,
        scheduled_experiment: ScheduledExperiment,
        rt_execution_info: RtExecutionInfo,
    ):
        initialization = get_initialization_by_device_uid(
            scheduled_experiment.recipe, self.uid
        )
        if initialization is None:
            return

        if scheduled_experiment.chunk_count is not None:
            raise LabOneQControllerException(
                f"{self.dev_repr}: Pipeliner is not supported by the device."
            )

        for output in initialization.outputs:
            self._warn_for_unsupported_param(
                output.gains is None, "correction_matrix", output.channel
            )
            if output.port_delay is not None:
                if output.port_delay != 0:
                    raise LabOneQControllerException(
                        f"{self.dev_repr}'s output does not support port delay"
                    )
                _logger.debug(
                    "%s's output port delay should be set to None, not 0",
                    self.dev_repr,
                )
            if output.range is not None:
                self._validate_range(output, is_out=True)

        for dev_input in initialization.inputs or []:
            if dev_input.range is None:
                continue
            self._validate_range(dev_input, is_out=False)

        # Validate average count
        # TODO(2K): Calculation of this_device_has_acquires duplicates the logic in
        # _validate_scheduled_experiment + _calculate_awg_configs. Cleanup after improving recipe.
        recipe = scheduled_experiment.recipe
        assert recipe is not None
        this_device_signals = {
            i.signal_id
            for i in recipe.integrator_allocations
            if i.device_id == self.uid
        }
        this_device_has_acquires = any(
            (this_device_signals & set(a.keys())) for a in recipe.simultaneous_acquires
        )
        averages = rt_execution_info.effective_averages
        if (
            this_device_has_acquires
            and not rt_execution_info.is_raw_acquisition
            and rt_execution_info.effective_averaging_mode != AveragingMode.SINGLE_SHOT
        ):
            if averages > MAX_AVERAGES_RESULT_LOGGER:
                raise LabOneQControllerException(
                    f"Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_RESULT_LOGGER}"
                )
            if averages & (averages - 1):
                raise LabOneQControllerException(
                    f"Number of averages {averages} must be a power of 2"
                )
        if rt_execution_info.is_raw_acquisition:
            if averages > MAX_AVERAGES_SCOPE:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Number of averages {averages} exceeds the allowed maximum {MAX_AVERAGES_SCOPE}"
                )
            # TODO(2K): Validate this at recipe processing stage (currently in _configure_input_monitor)
            # if awg_config.raw_acquire_length is None:
            #     raise LabOneQControllerException(
            #         f"{self.dev_repr}: Unknown acquire length for RAW acquisition."
            #     )

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
        # For non-desktop, always use external clock,
        # for desktop - internal is the default (force_internal is None),
        # but allow override to external.
        self._use_internal_clock = self.is_desktop() and (force_internal is not False)

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

    async def start_execution(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        for awg_index in recipe_data.allocated_awgs(self.uid):
            nc.add(f"awgs/{awg_index}/enable", 1, cache=False)
        await self.set_async(nc)

    def conditions_for_execution_ready(
        self, recipe_data: RecipeData
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in recipe_data.allocated_awgs(self.uid):
            conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                1,
                f"AWG {awg_index} didn't start.",
            )
        return conditions

    def conditions_for_execution_done(
        self, recipe_data: RecipeData
    ) -> dict[str, tuple[Any, str]]:
        conditions: dict[str, tuple[Any, str]] = {}
        for awg_index in recipe_data.allocated_awgs(self.uid):
            conditions[f"/{self.serial}/awgs/{awg_index}/enable"] = (
                0,
                f"AWG {awg_index} didn't stop. Missing start trigger? Check DIO.",
            )
        return conditions

    async def apply_initialization(self, recipe_data: RecipeData):
        uhfqa_recipe_data = recipe_data.device_settings[self.uid].uhfqacore
        await for_each(
            self.all_channels(),
            UHFQAAwg.apply_initialization,
            uhfqa_recipe_data=uhfqa_recipe_data,
        )

    async def _set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        nc = NodeCollector(base=f"/{self.serial}/")

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
                ch_repr=f"{self.dev_repr}:ch{ch}",
                delay=measurement_delay,
                sample_frequency_hz=SAMPLE_FREQUENCY_HZ,
                granularity_samples=DELAY_NODE_GRANULARITY_SAMPLES,
                max_node_delay_samples=DELAY_NODE_MAX_SAMPLES,
            )

            nc.add("qas/0/delay", measurement_delay_rounded)

        await self.set_async(nc)

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

    async def set_before_awg_upload(self, recipe_data: RecipeData):
        acquisition_type = recipe_data.rt_execution_info.acquisition_type
        if acquisition_type == AcquisitionType.SPECTROSCOPY_IQ:
            nc = self._configure_spectroscopy_mode_nodes()
        else:
            nc = self._configure_standard_mode_nodes(
                acquisition_type, self.uid, recipe_data
            )
        await self.set_async(nc)

    async def set_after_awg_upload(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        initialization = recipe_data.get_initialization(self.uid)
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
            nc.add(f"sigins/{dev_input.channel}/range", dev_input.range)

        await self.set_async(nc)

    async def configure_trigger(self, recipe_data: RecipeData):
        device_recipe_data = recipe_data.device_settings[self.uid]
        nc = NodeCollector(base=f"/{self.serial}/")

        # Loop over at least AWG instance to cover the case that the instrument is only used as a
        # communication proxy. Some of the nodes on the AWG branch are needed to get proper
        # communication between HDAWG and UHFQA.
        for awg_index in device_recipe_data.allocated_awgs(default_awg=0):
            nc.add(f"awgs/{awg_index}/dio/strobe/index", 16)
            nc.add(f"awgs/{awg_index}/dio/strobe/slope", 0)
            nc.add(f"awgs/{awg_index}/dio/valid/polarity", 2)
            nc.add(f"awgs/{awg_index}/dio/valid/index", 16)

        if self.is_desktop():
            nc.add("dios/0/mode", 0)
            nc.add("dios/0/drive", 0)
            nc.add("dios/0/extclk", 0x2)
            nc.add("awgs/0/auxtriggers/0/channel", 0)
            nc.add("awgs/0/auxtriggers/0/slope", 1)
        else:
            nc.add("dios/0/mode", 4)
            nc.add("dios/0/drive", 0x3)
            nc.add("dios/0/extclk", 0x2)

        for trigger_index in (0, 1):
            nc.add(f"triggers/out/{trigger_index}/delay", 0.0)
            nc.add(f"triggers/out/{trigger_index}/drive", 1)
            nc.add(f"triggers/out/{trigger_index}/source", 32 + trigger_index)

        await self.set_async(nc)

    async def on_experiment_begin(self, recipe_data: RecipeData):
        nodes = [
            *(
                self._result_node_integrator(result_index)
                for result_index in range(self._integrators)
            ),
            self._result_node_monitor(0),
            self._result_node_monitor(1),
        ]
        await _gather(
            super().on_experiment_begin(recipe_data=recipe_data),
            *(self._subscriber.subscribe(self._api, node) for node in nodes),
        )

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
        self, result_index, num_results, averages_divider: int, timeout_s: float
    ):
        result_path = self._result_node_integrator(result_index)
        try:
            integrator_result = await self._subscriber.get_result(
                result_path, timeout_s=timeout_s
            )
            self._check_result(
                node_val=integrator_result.vector,
                num_results=num_results,
                ch_repr=self._ch_repr_readout(result_index),
            )
            # Not dividing by averages_divider - it appears poll data is already divided.
            return integrator_result.vector[0:num_results]
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{self._ch_repr_readout(result_index)}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            return np.array([], dtype=np.complex128)

    async def get_measurement_data(
        self,
        channel: int,
        recipe_data: RecipeData,
        result_indices: list[int],
        num_results: int,
        hw_averages: int,
    ) -> RawReadoutData:
        # In the async execution model, result waiting starts as soon as execution begins,
        # so the execution time must be included when calculating the result retrieval timeout.
        _, guarded_wait_time = get_execution_time(recipe_data)
        # TODO(2K): set timeout based on timeout_s from connect
        timeout_s = 5 + guarded_wait_time

        rt_execution_info = recipe_data.rt_execution_info
        averages_divider = (
            1
            if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION
            else hw_averages
        )
        assert len(result_indices) <= 2
        if len(result_indices) == 1:
            data = await self._get_integrator_measurement_data(
                result_indices[0], num_results, averages_divider, timeout_s
            )
            return RawReadoutData(data)
        else:
            in_phase, quadrature = await _gather(
                self._get_integrator_measurement_data(
                    result_indices[0], num_results, averages_divider, timeout_s
                ),
                self._get_integrator_measurement_data(
                    result_indices[1], num_results, averages_divider, timeout_s
                ),
            )
            return RawReadoutData(
                np.array(
                    [complex(real, imag) for real, imag in zip(in_phase, quadrature)]
                )
            )

    async def _get_input_monitor_data(
        self, ch: int, acquire_length: int, timeout_s: float
    ):
        result_path = self._result_node_monitor(ch)
        try:
            monitor_result = await self._subscriber.get_result(
                result_path, timeout_s=timeout_s
            )
            self._check_result(
                node_val=monitor_result.vector,
                num_results=acquire_length,
                ch_repr=self._ch_repr_monitor(ch),
            )
            # Truncate returned vectors to the expected length -> hotfix for GCE-681
            return monitor_result.vector[0:acquire_length]
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{self._ch_repr_monitor(ch)}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            return np.array([], dtype=np.complex128)

    async def get_raw_data(
        self, channel: int, acquire_length: int, acquires: int | None, timeout_s: float
    ) -> RawReadoutData:
        ch0, ch1 = await _gather(
            self._get_input_monitor_data(0, acquire_length, timeout_s),
            self._get_input_monitor_data(1, acquire_length, timeout_s),
        )
        return RawReadoutData(
            np.array([[complex(real, imag) for real, imag in zip(ch0, ch1)]])
        )
