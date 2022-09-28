# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List
from laboneq.controller.recipe_1_4_0 import Initialization

from laboneq.controller.recipe_processor import DeviceRecipeData
from laboneq.controller.recipe_enums import ReferenceClockSource
from .device_zi import DeviceZI

from ..communication import (
    DaqNodeAction,
    DaqNodeSetAction,
    DaqNodeWaitAction,
    CachingStrategy,
)

from laboneq.controller.util import LabOneQControllerException

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1


class DevicePQSC(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "PQSC"
        self.dev_opts = []

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

    def collect_conditions_to_close_loop(self, acquisition_units):
        return [
            DaqNodeWaitAction(
                self._daq,
                f"/{self.serial}/execution/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        ]

    def collect_trigger_configuration_nodes(self, initialization):
        # Ensure ZSync links are established
        # TODO(2K): This is rather a hotfix, waiting to be done in parallel for all devices with subscription / poll
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

    def collect_follower_configuration_nodes(self, initialization):
        raise LabOneQControllerException("PQSC cannot be configured as follower")

    def configure_as_leader(self, initialization):
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

        clock_source = initialization.config.reference_clock_source
        if clock_source and clock_source.value == ReferenceClockSource.INTERNAL.value:
            self._switch_reference_clock(
                source=REFERENCE_CLOCK_SOURCE_INTERNAL, expected_freqs=None
            )
        else:
            self._switch_reference_clock(
                source=REFERENCE_CLOCK_SOURCE_EXTERNAL, expected_freqs=10e6
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
