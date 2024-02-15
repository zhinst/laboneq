# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any
from laboneq.controller.communication import DaqNodeSetAction, DaqWrapper
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import (
    DeviceQualifier,
    DeviceZI,
)
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.data.recipe import Initialization, NtStepKey
from laboneq.controller.attribute_value_tracker import (
    DeviceAttributesView,
)


_logger = logging.getLogger(__name__)


class DevicePRETTYPRINTER(DeviceZI):
    def __init__(self, device_qualifier: DeviceQualifier, daq: DaqWrapper):
        super().__init__(device_qualifier=device_qualifier, daq=daq)
        self._device_class = 0x1

    async def connect(self, emulator_state: Any):
        _logger.info(
            "%s: Connected to %s",
            self.dev_repr,
            self.device_qualifier.options.serial,
        )

    def disconnect(self):
        pass

    async def prepare_artifacts(
        self,
        recipe_data: RecipeData,
        rt_section_uid: str,
        initialization: Initialization,
        awg_index: int,
        nt_step: NtStepKey,
    ) -> tuple[
        DeviceZI, list[DaqNodeSetAction], list[DaqNodeSetAction], dict[str, Any]
    ]:
        return self, [], [], {}

    async def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ):
        return []

    async def collect_osc_initialization_nodes(self) -> list[DaqNodeSetAction]:
        return []

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> NodeCollector:
        return NodeCollector()

    async def collect_execution_nodes(self, *args, **kwargs):
        return []

    async def collect_reset_nodes(self) -> list[DaqNodeSetAction]:
        return []

    async def fetch_errors(self):
        return []
