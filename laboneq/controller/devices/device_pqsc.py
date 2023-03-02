# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
)
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.devices.zi_node_monitor import (
    Command,
    Condition,
    NodeControlBase,
    Response,
)
from laboneq.controller.recipe_1_4_0 import Initialization
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.acquisition_type import AcquisitionType

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1


class DevicePQSC(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "PQSC"
        self.dev_opts = []
        self._use_internal_clock = False

    def _nodes_to_monitor_impl(self) -> List[str]:
        nodes = [node.path for node in self.clock_source_control_nodes()]
        nodes.append(f"/{self.serial}/execution/enable")
        return nodes

    def update_clock_source(self, force_internal: Optional[bool]):
        self._use_internal_clock = force_internal is True

    def clock_source_control_nodes(self) -> List[NodeControlBase]:
        source = (
            REFERENCE_CLOCK_SOURCE_INTERNAL
            if self._use_internal_clock
            else REFERENCE_CLOCK_SOURCE_EXTERNAL
        )
        expected_freq = None if self._use_internal_clock else 10e6
        return [
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/freq", expected_freq
            ),
            Command(f"/{self.serial}/system/clocks/referenceclock/in/source", source),
            Response(
                f"/{self.serial}/system/clocks/referenceclock/in/sourceactual", source
            ),
            Response(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
        ]

    def collect_output_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        return []

    def collect_execution_nodes(self):
        self._logger.debug("Starting execution...")
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/execution/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/triggers/out/0/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
        ]

    def conditions_for_execution_done(
        self, acquisition_type: AcquisitionType
    ) -> Dict[str, Any]:
        return {f"/{self.serial}/execution/enable": 0}

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> List[DaqNodeAction]:
        # Ensure ZSync links are established
        # TODO(2K): This is rather a hotfix, waiting to be done in parallel for all devices with
        # subscription / poll
        # TODO(2K): Verify also the downlink device serial (.../connection/serial) matches
        for port, _ in self._downlinks.items():
            self._wait_for_node(
                f"/{self.serial}/{port.lower()}/connection/status", 2, timeout=10
            )

        self._logger.debug(
            "%s: Configuring holdoff time: %f s.",
            self.dev_repr,
            initialization.config.holdoff,
        )
        self._logger.debug(
            "%s: Configuring repetitions: %d.",
            self.dev_repr,
            initialization.config.repetitions,
        )
        nodes_to_configure_triggers = []

        nodes_to_configure_triggers.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/execution/holdoff",
                initialization.config.holdoff,
            )
        )
        nodes_to_configure_triggers.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/execution/repetitions",
                initialization.config.repetitions,
            )
        )

        return nodes_to_configure_triggers

    def collect_follower_configuration_nodes(
        self, initialization: Initialization.Data
    ) -> List[DaqNodeAction]:
        raise LabOneQControllerException("PQSC cannot be configured as follower")

    def configure_as_leader(self, initialization: Initialization.Data):
        self._logger.debug("%s: Configuring as leader...", self.dev_repr)
        self._logger.debug("%s: Enabling reference clock...", self.dev_repr)

        self._logger.debug(
            "%s: Setting reference clock frequency to %d MHz...",
            self.dev_repr,
            initialization.config.reference_clock,
        )

        self._daq.batch_set(
            [
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/clocks/referenceclock/out/enable",
                    1,
                )
            ]
        )

        self._daq.batch_set(
            [
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/system/clocks/referenceclock/out/freq",
                    initialization.config.reference_clock,
                )
            ]
        )

    def initialize_sweep_setting(self, setting):
        raise LabOneQControllerException("PQSC doesn't support sweeping")
