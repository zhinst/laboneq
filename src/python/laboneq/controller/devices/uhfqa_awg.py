# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.controller.attribute_value_tracker import DeviceAttributesView
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
    ResponseWaiterAsync,
    _gather,
)
from laboneq.controller.devices.core_base import CoreBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import RawReadoutData
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    HWModulation,
    RecipeData,
    RtExecutionInfo,
    UHFQARecipeData,
    get_elf,
    get_execution_time,
    get_wave,
    get_weights_info,
    prepare_waves,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IntegratorAllocation, NtStepKey
from laboneq.data.scheduled_experiment import ArtifactsCodegen

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray

_logger = logging.getLogger(__name__)


@dataclass
class QAOutputNodes:
    output_on: str


class QAOutput:
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        channel: int,
        repr_base: str,
    ):
        self._api = api
        self._subscriber = subscriber
        self._device_uid = device_uid
        self._serial = serial
        self._channel = channel
        self._unit_repr = f"{repr_base}:ch{channel}"
        self.nodes = QAOutputNodes(
            output_on=f"/{serial}/sigouts/{channel}/on",
        )

    async def disable_output(self, outputs: set[int], invert: bool):
        if (self._channel in outputs) != invert:
            await self._api.set_parallel(
                NodeCollector.one(self.nodes.output_on, 0, cache=False)
            )

    async def configure(self, uhfqa_recipe_data: UHFQARecipeData):
        ch_recipe_data = uhfqa_recipe_data.outputs[self._channel]
        nc = NodeCollector(base=f"/{self._serial}/sigouts/{self._channel}/")

        nc.add("on", 1 if ch_recipe_data.enable else 0)
        if ch_recipe_data.enable:
            nc.add("imp50", 1)
        if ch_recipe_data.offset is not None:
            nc.add("offset", ch_recipe_data.offset)

        # the following is needed so that in spectroscopy mode, pulse lengths are correct
        # TODO(2K): Why 2 enables per sigout, but only one is used?
        nc.add(f"enables/{self._channel}", 1)

        if ch_recipe_data.range is not None:
            nc.add("range", ch_recipe_data.range)

        nc.add_absolute(
            f"/{self._serial}/awgs/0/outputs/{self._channel}/mode",
            0 if ch_recipe_data.hw_modulation == HWModulation.OFF else 1,
        )

        await self._api.set_parallel(nc)


@dataclass
class UHFQAAwgNodes:
    awg_enable: str
    awg_sequencer_status: str
    readout_result_wave: list[str]
    monitor_result_wave: list[str]


def _check_result(node_val: NumPyArray, num_results: int, ch_repr: str):
    num_samples = len(node_val)
    if num_samples != num_results:
        _logger.error(
            f"{ch_repr}: The number of measurements acquired ({num_samples}) "
            f"does not match the number of measurements defined ({num_results}). "
            "Possibly the time between measurements within a loop is too short, "
            "or the measurement was not started."
        )


class UHFQAAwg(CoreBase):
    def __init__(
        self,
        *,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        repr_base: str,
        integrators: int,
    ):
        super().__init__(
            api=api,
            subscriber=subscriber,
            device_uid=device_uid,
            serial=serial,
            core_index=0,
        )
        self._node_base = f"/{serial}/"
        self._unit_repr = repr_base
        self._integrators = integrators
        self._outputs: list[QAOutput] = [
            QAOutput(
                api,
                subscriber,
                device_uid,
                serial,
                channel=ch,
                repr_base=self._unit_repr,
            )
            for ch in range(2)
        ]
        self.nodes = UHFQAAwgNodes(
            awg_enable=f"/{self._serial}/awgs/0/enable",
            awg_sequencer_status=f"/{self._serial}/awgs/0/sequencer/status",
            readout_result_wave=[
                f"/{self._serial}/qas/0/result/data/{i}/wave"
                for i in range(integrators)
            ],
            monitor_result_wave=[
                f"/{self._serial}/qas/0/monitor/inputs/{0}/wave",
                f"/{self._serial}/qas/0/monitor/inputs/{1}/wave",
            ],
        )

    async def disable_output(self, outputs: set[int], invert: bool):
        await _gather(
            *[output.disable_output(outputs, invert) for output in self._outputs]
        )

    def allocate_resources(self):
        pass

    async def _configure_awg_core(self):
        await self._api.set_parallel(
            NodeCollector.one(f"/{self._serial}/awgs/0/single", 1)
        )

    def subscribe_nodes(self) -> list[str]:
        return [
            *self.nodes.readout_result_wave,
            *self.nodes.monitor_result_wave,
        ]

    async def apply_initialization(self, uhfqa_recipe_data: UHFQARecipeData):
        await _gather(
            self._configure_awg_core(),
            self._outputs[0].configure(uhfqa_recipe_data=uhfqa_recipe_data),
            self._outputs[1].configure(uhfqa_recipe_data=uhfqa_recipe_data),
        )

    def _configure_standard_mode_nodes(
        self,
        acquisition_type: AcquisitionType,
        device_uid: str,
        recipe_data: RecipeData,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self._serial}/")

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
                        f"{self._unit_repr}: Internal error - expected 2 integrators for signal "
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
                        f"{self._unit_repr}: Internal error - expected 1 integrator for signal "
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
        nc = NodeCollector(base=f"/{self._serial}/")
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

    async def _set_before_awg_upload(self, recipe_data: RecipeData):
        acquisition_type = recipe_data.rt_execution_info.acquisition_type
        if acquisition_type == AcquisitionType.SPECTROSCOPY_IQ:
            nc = self._configure_spectroscopy_mode_nodes()
        else:
            nc = self._configure_standard_mode_nodes(
                acquisition_type, self._device_uid, recipe_data
            )

        initialization = recipe_data.get_initialization(self._device_uid)
        inputs = initialization.inputs
        if len(initialization.measurements) > 0:
            [measurement] = initialization.measurements
            nc.add("qas/0/integration/length", measurement.length)
            nc.add("qas/0/integration/trigger/channel", 7)

        for dev_input in inputs or []:
            if dev_input.range is None:
                continue
            nc.add(f"sigins/{dev_input.channel}/range", dev_input.range)

        await self._api.set_parallel(nc)

    async def load_awg_program(self, recipe_data: RecipeData, nt_step: NtStepKey):
        await self._set_before_awg_upload(recipe_data)
        artifacts = recipe_data.get_artifacts(ArtifactsCodegen)

        elf_nodes = NodeCollector()
        wf_nodes = NodeCollector()
        upload_ready_conditions: dict[str, Any] = {}

        rt_exec_step = next(
            (
                r
                for r in recipe_data.recipe.realtime_execution_init
                if r.device_id == self._device_uid
                and r.awg_index == 0
                and r.nt_step == nt_step
            ),
            None,
        )

        # Todo (PW): This currently needlessly reconfigures the acquisition in every
        #  NT step. The acquisition parameters really are constant across NT steps,
        #  we only care about re-enabling the result logger.
        wf_nodes.extend(self.configure_acquisition(recipe_data))

        if rt_exec_step is not None:
            seqc_elf = get_elf(artifacts, rt_exec_step.program_ref)
            if seqc_elf is not None:
                elf_nodes.add(
                    f"/{self._serial}/awgs/0/elf/data",
                    seqc_elf,
                    cache=False,
                    filename=rt_exec_step.program_ref,
                )
                upload_ready_conditions.update({f"/{self._serial}/awgs/0/ready": 1})

            waves = prepare_waves(artifacts, rt_exec_step.wave_indices_ref)
            if waves is not None:
                for wave in waves:
                    wf_nodes.add(
                        path=f"/{self._serial}/awgs/{0}/waveform/waves/{wave.index}",
                        value=wave.samples,
                        cache=False,
                        filename=wave.name,
                    )

            wf_nodes.extend(
                # TODO(2K): Cleanup arguments to prepare_upload_all_integration_weights
                self.prepare_upload_all_integration_weights(
                    artifacts,
                    recipe_data.recipe.integrator_allocations,
                    rt_exec_step.kernel_indices_ref,
                )
            )

        rw = ResponseWaiterAsync(api=self._api, dev_repr=self._unit_repr, timeout_s=10)
        rw.add_nodes(upload_ready_conditions)
        await rw.prepare()
        await self._api.set_parallel(elf_nodes)
        await rw.wait()
        await self._api.set_parallel(wf_nodes)

    def configure_acquisition(self, recipe_data: RecipeData) -> NodeCollector:
        rt_execution_info = recipe_data.rt_execution_info
        awg_config = recipe_data.awg_configs[AwgKey(self._device_uid, 0)]
        nc = NodeCollector()
        nc.extend(
            self._configure_result_logger(
                awg_config=awg_config,
                rt_execution_info=rt_execution_info,
            )
        )
        nc.extend(
            self._configure_input_monitor(
                awg_config=awg_config,
                rt_execution_info=rt_execution_info,
            )
        )
        return nc

    def _configure_result_logger(
        self,
        awg_config: AwgConfig,
        rt_execution_info: RtExecutionInfo,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self._serial}/")
        if awg_config.result_length is None:
            return nc  # this instrument is unused for acquiring results
        enable = not rt_execution_info.is_raw_acquisition
        if enable:
            nc.add("qas/0/result/length", awg_config.result_length)
            nc.add("qas/0/result/averages", rt_execution_info.effective_averages)
            nc.add(
                "qas/0/result/mode",
                0
                if rt_execution_info.effective_averaging_mode == AveragingMode.CYCLIC
                else 1,
            )
            nc.add(
                "qas/0/result/source",
                1  # result source 'threshold'
                if rt_execution_info.acquisition_type == AcquisitionType.DISCRIMINATION
                else 2,  # result source 'rotation'
            )
            nc.add("qas/0/result/enable", 0)
            nc.add("qas/0/result/reset", 1, cache=False)
        nc.barrier()
        nc.add("qas/0/result/enable", 1 if enable else 0)
        nc.barrier()
        return nc

    def _configure_input_monitor(
        self,
        awg_config: AwgConfig,
        rt_execution_info: RtExecutionInfo,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self._serial}/")
        enable = rt_execution_info.is_raw_acquisition
        if enable:
            acquire_length = awg_config.raw_acquire_length
            # TODO(2K): Validate this at recipe processing stage
            if acquire_length is None:
                raise LabOneQControllerException(
                    f"{self._unit_repr}: Unknown acquire length for RAW acquisition."
                )
            nc.add("qas/0/monitor/length", acquire_length)
            nc.add("qas/0/monitor/averages", rt_execution_info.effective_averages)
            nc.add("qas/0/monitor/enable", 0)  # todo: barrier needed?
            nc.add("qas/0/monitor/reset", 1, cache=False)
        nc.add("qas/0/monitor/enable", 1 if enable else 0)
        return nc

    def prepare_upload_all_integration_weights(
        self,
        artifacts: ArtifactsCodegen,
        integrator_allocations: list[IntegratorAllocation],
        kernel_ref: str | None,
    ) -> NodeCollector:
        nc = NodeCollector(base=f"/{self._serial}/")

        weights_info = get_weights_info(artifacts, kernel_ref)
        for signal_id, weight_names in weights_info.items():
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

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        raise NotImplementedError

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {
            self.nodes.awg_sequencer_status: (
                4,
                f"AWG {self._core_index} didn't start.",
            )
        }

    async def start_execution(self, with_pipeliner: bool):
        raise NotImplementedError

    def conditions_for_execution_done(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {
            self.nodes.awg_enable: (
                0,
                f"AWG {self._core_index} didn't stop. Missing start trigger? Check DIO.",
            )
        }

    async def get_measurement_data(
        self,
        *,
        recipe_data: RecipeData,
        num_results: int,
        hw_averages: int,
    ) -> list[RawReadoutData]:
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
        all_data = await _gather(
            *[
                self._get_integrator_measurement_data(
                    integrator, num_results, averages_divider, timeout_s
                )
                for integrator in range(self._integrators)
            ]
        )
        return [RawReadoutData(data) for data in all_data]

    async def _get_integrator_measurement_data(
        self, result_index, num_results, averages_divider: int, timeout_s: float
    ):
        result_path = self.nodes.readout_result_wave[result_index]
        ch_repr_readout = f"{self._unit_repr}:readout{result_index}"
        try:
            integrator_result = await self._subscriber.get_result(
                result_path, timeout_s=timeout_s
            )
            _check_result(
                node_val=integrator_result.vector,
                num_results=num_results,
                ch_repr=ch_repr_readout,
            )
            # Not dividing by averages_divider - it appears poll data is already divided.
            return integrator_result.vector[0:num_results]
        except (TimeoutError, asyncio.TimeoutError):
            _logger.error(
                f"{ch_repr_readout}: Failed to receive a result from {result_path} within {timeout_s} seconds."
            )
            return np.array([], dtype=np.complex128)
