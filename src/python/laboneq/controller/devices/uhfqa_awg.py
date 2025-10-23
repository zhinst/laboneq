# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Any

from laboneq.controller.attribute_value_tracker import DeviceAttributesView
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
    ResponseWaiterAsync,
    _gather,
)
from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.recipe_processor import (
    AwgConfig,
    AwgKey,
    HWModulation,
    RecipeData,
    RtExecutionInfo,
    UHFQARecipeData,
    get_elf,
    get_wave,
    get_weights_info,
    prepare_waves,
)
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.data.recipe import IntegratorAllocation, NtStepKey
from laboneq.data.scheduled_experiment import ArtifactsCodegen
import numpy as np


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


class UHFQAAwg(ChannelBase):
    def __init__(
        self,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        repr_base: str,
    ):
        super().__init__(api, subscriber, device_uid, serial, 0)
        self._node_base = f"/{serial}/"
        self._unit_repr = repr_base
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

    def _disable_output(self) -> NodeCollector:
        raise NotImplementedError

    def allocate_resources(self):
        pass

    async def _configure_awg_core(self):
        await self._api.set_parallel(
            NodeCollector.one(f"/{self._serial}/awgs/0/single", 1)
        )

    async def apply_initialization(self, uhfqa_recipe_data: UHFQARecipeData):
        await _gather(
            self._configure_awg_core(),
            self._outputs[0].configure(uhfqa_recipe_data=uhfqa_recipe_data),
            self._outputs[1].configure(uhfqa_recipe_data=uhfqa_recipe_data),
        )

    async def load_awg_program(self, recipe_data: RecipeData, nt_step: NtStepKey):
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

    async def start_execution(self, with_pipeliner: bool):
        raise NotImplementedError
