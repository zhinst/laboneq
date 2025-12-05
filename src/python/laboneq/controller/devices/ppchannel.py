# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from laboneq.controller.attribute_value_tracker import DeviceAttributesView
from laboneq.controller.devices.async_support import (
    AsyncSubscriber,
    InstrumentConnection,
)
from laboneq.controller.devices.core_base import CoreBase
from laboneq.controller.recipe_processor import RecipeData
from laboneq.data.recipe import NtStepKey


class PPChannel(CoreBase):
    def __init__(
        self,
        *,
        api: InstrumentConnection,
        subscriber: AsyncSubscriber,
        device_uid: str,
        serial: str,
        core_index: int,
    ):
        super().__init__(
            api=api,
            subscriber=subscriber,
            device_uid=device_uid,
            serial=serial,
            core_index=core_index,
        )

    async def disable_output(self, outputs: set[int], invert: bool):
        pass

    def allocate_resources(self):
        pass

    async def load_awg_program(
        self,
        recipe_data: RecipeData,
        nt_step: NtStepKey,
    ):
        pass

    async def set_nt_step_nodes(
        self, recipe_data: RecipeData, attributes: DeviceAttributesView
    ):
        pass

    def collect_warning_nodes(self) -> list[tuple[str, str]]:
        return []

    def conditions_for_execution_ready(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {
            f"/{self._serial}/ppchannels/{self._core_index}/sweeper/enable": (
                1,
                f"Sweeper {self._core_index} didn't start.",
            )
        }

    async def start_execution(self, with_pipeliner: bool):
        pass

    def conditions_for_execution_done(
        self, with_pipeliner: bool
    ) -> dict[str, tuple[Any, str]]:
        return {
            f"/{self._serial}/ppchannels/{self._core_index}/sweeper/enable": (
                0,
                f"Sweeper on channel {self._core_index} didn't stop. Check trigger connection.",
            )
        }
