# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from laboneq.controller.communication import DaqNodeAction, DaqWrapper
from laboneq.controller.devices.device_zi import DeviceQualifier, DeviceZI
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.data.recipe import Initialization
from laboneq.controller.attribute_value_tracker import (
    DeviceAttributesView,
)
from laboneq.compiler.workflow.compiler_output import ArtifactsPrettyPrinter


_logger = logging.getLogger(__name__)


class DevicePRETTYPRINTER(DeviceZI):
    def __init__(self, device_qualifier: DeviceQualifier, daq: DaqWrapper):
        super().__init__(device_qualifier=device_qualifier, daq=daq)
        self._device_class = 0x1

    async def connect(self):
        _logger.info(
            "%s: Connected to %s",
            self.dev_repr,
            self.device_qualifier.options.serial,
        )

    def disconnect(self):
        pass

    async def prepare_artifacts(
        self,
        artifacts: ArtifactsPrettyPrinter | dict[int, ArtifactsPrettyPrinter],
        channel: str,
        instructions_ref: str,
        waves_ref: str,
    ):
        pass

    async def collect_initialization_nodes(
        self,
        device_recipe_data: DeviceRecipeData,
        initialization: Initialization,
        recipe_data: RecipeData,
    ):
        return []

    async def collect_osc_initialization_nodes(self) -> list[DaqNodeAction]:
        return []

    def collect_prepare_nt_step_nodes(
        self, attributes: DeviceAttributesView, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        return []

    async def collect_execution_nodes(self, *args, **kwargs):
        return []

    async def collect_reset_nodes(self) -> list[DaqNodeAction]:
        return []

    async def fetch_errors(self):
        return []
