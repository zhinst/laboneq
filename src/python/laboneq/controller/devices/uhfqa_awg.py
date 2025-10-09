# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.controller.attribute_value_tracker import DeviceAttributesView
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
    _gather,
)
from laboneq.controller.devices.channel_base import ChannelBase
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.recipe_processor import (
    HWModulation,
    RecipeData,
    UHFQARecipeData,
)
from laboneq.data.recipe import NtStepKey


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
        raise NotImplementedError

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        raise NotImplementedError

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        raise NotImplementedError

    async def start_execution(self, with_pipeliner: bool):
        raise NotImplementedError
