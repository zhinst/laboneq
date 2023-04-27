# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Any

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

_logger = logging.getLogger(__name__)

REFERENCE_CLOCK_SOURCE_INTERNAL = 0
REFERENCE_CLOCK_SOURCE_EXTERNAL = 1


class DevicePQSC(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dev_type = "PQSC"
        self.dev_opts = []
        self._use_internal_clock = False

    def _nodes_to_monitor_impl(self) -> list[str]:
        nodes = super()._nodes_to_monitor_impl()
        nodes.append(f"/{self.serial}/execution/enable")
        return nodes

    def update_clock_source(self, force_internal: bool | None):
        self._use_internal_clock = force_internal is True

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
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

    def collect_initialization_nodes(
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization.Data
    ) -> list[DaqNodeAction]:
        return []

    def configure_feedback(self, recipe_data: RecipeData) -> list[DaqNodeAction]:
        # TODO(2K): Code duplication with Controller._wait_execution_to_stop
        # Make this mandatory in the recipe instead.
        min_wait_time = recipe_data.recipe.experiment.total_execution_time
        if min_wait_time is None:
            min_wait_time = 10.0
        # This is required because PQSC is only receiving the feedback events
        # during the holdoff time, even for a single trigger.
        feedback_actions = [
            DaqNodeSetAction(
                self.daq, f"/{self.serial}/execution/holdoff", min_wait_time
            )
        ]
        for port, (follower_uid, follower_ref) in self._downlinks.items():
            [p_kind, p_addr] = port.split("/")
            follower = follower_ref()
            if p_kind != "ZSYNCS" or follower is None:
                continue
            for awg_key, awg_config in recipe_data.awg_configs.items():
                if (
                    awg_key.device_uid != follower_uid
                    or awg_config.source_feedback_register is None
                ):
                    continue  # Only consider devices receiving feedback from PQSC
                zsync_base = f"/{self.serial}/zsyncs/{p_addr}/output/registerbank"
                feedback_actions.append(
                    DaqNodeSetAction(self.daq, f"{zsync_base}/enable", 1)
                )
                bit_base = f"{zsync_base}/sources/{awg_config.zsync_bit}"
                feedback_actions.extend(
                    [
                        DaqNodeSetAction(self.daq, f"{bit_base}/enable", 1),
                        DaqNodeSetAction(
                            self.daq,
                            f"{bit_base}/register",
                            awg_config.source_feedback_register,
                        ),
                        DaqNodeSetAction(
                            self.daq,
                            f"{bit_base}/index",
                            awg_config.feedback_register_bit,
                        ),
                    ]
                )
        return feedback_actions

    def collect_execution_nodes(self):
        _logger.debug("Starting execution...")
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
    ) -> dict[str, Any]:
        return {f"/{self.serial}/execution/enable": 0}

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization.Data, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        # Ensure ZSync links are established
        # TODO(2K): This is rather a hotfix, waiting to be done in parallel for all devices with
        # subscription / poll
        # TODO(2K): Verify also the downlink device serial (.../connection/serial) matches
        for port in self._downlinks:
            self._wait_for_node(
                f"/{self.serial}/{port.lower()}/connection/status", 2, timeout=10
            )

        _logger.debug(
            "%s: Configuring holdoff time: %f s.",
            self.dev_repr,
            initialization.config.holdoff,
        )
        _logger.debug(
            "%s: Configuring repetitions: %d.",
            self.dev_repr,
            initialization.config.repetitions,
        )
        nodes_to_configure_triggers = []

        # TODO(2K): 'recipe.initialization.config.holdoff' is hard-coded to 0 in the compiler,
        # but the PQSC hold-off needs to be set for the feedback, see 'configure_feedback()'.
        # Resolve this uncertainty by either dropping 'recipe.initialization.config.holdoff',
        # or setting it to the right value in the compiler, but then it may duplicate
        # 'recipe.experiment.total_execution_time'.

        # nodes_to_configure_triggers.append(
        #     DaqNodeSetAction(
        #         self._daq,
        #         f"/{self.serial}/execution/holdoff",
        #         initialization.config.holdoff,
        #     )
        # )

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
    ) -> list[DaqNodeAction]:
        raise LabOneQControllerException("PQSC cannot be configured as follower")

    def configure_as_leader(self, initialization: Initialization.Data):
        _logger.debug("%s: Configuring as leader...", self.dev_repr)
        _logger.debug("%s: Enabling reference clock...", self.dev_repr)

        _logger.debug(
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
