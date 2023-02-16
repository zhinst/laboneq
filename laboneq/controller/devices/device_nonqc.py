# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List

from laboneq.controller.communication import DaqNodeAction
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.recipe_1_4_0 import Initialization
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData


class DeviceNonQC(DeviceZI):
    def is_leader(self):
        return False

    def is_follower(self):
        return False

    def is_standalone(self):
        return False

    def collect_follower_configuration_nodes(
        self, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        return []

    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        return []

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> List[DaqNodeAction]:
        return []

    def configure_as_leader(self, initialization: Initialization.Data):
        pass

    def check_errors(self):
        pass

    def collect_reset_nodes(self) -> List[DaqNodeAction]:
        return []
