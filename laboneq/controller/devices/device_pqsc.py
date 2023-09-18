# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import IntEnum

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
from laboneq.controller.recipe_processor import DeviceRecipeData, RecipeData
from laboneq.controller.versioning import SUPPORT_PRE_V23_06, LabOneVersion
from laboneq.data.recipe import Initialization

_logger = logging.getLogger(__name__)


class ReferenceClockSourcePQSC(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1


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
            ReferenceClockSourcePQSC.INTERNAL
            if self._use_internal_clock
            else ReferenceClockSourcePQSC.EXTERNAL
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
        self, device_recipe_data: DeviceRecipeData, initialization: Initialization
    ) -> list[DaqNodeAction]:
        return []

    def configure_feedback(self, recipe_data: RecipeData) -> list[DaqNodeAction]:
        min_wait_time = recipe_data.recipe.max_step_execution_time
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

                zsync_output = f"/{self.serial}/zsyncs/{p_addr}/output"
                zsync_base = f"{zsync_output}/registerbank"
                if SUPPORT_PRE_V23_06 and (
                    self.daq._dataserver_version < LabOneVersion.V_23_06
                ):
                    actions_to_enable_feedback = [
                        DaqNodeSetAction(self.daq, f"{zsync_base}/enable", 1)
                    ]
                else:
                    actions_to_enable_feedback = [
                        DaqNodeSetAction(self.daq, f"{zsync_output}/enable", 1),
                        DaqNodeSetAction(self.daq, f"{zsync_output}/source", 0),
                    ]

                feedback_actions.extend(actions_to_enable_feedback)
                reg_selector_base = (
                    f"{zsync_base}/sources/{awg_config.register_selector_index}"
                )
                feedback_actions.extend(
                    [
                        DaqNodeSetAction(self.daq, f"{reg_selector_base}/enable", 1),
                        DaqNodeSetAction(
                            self.daq,
                            f"{reg_selector_base}/register",
                            awg_config.source_feedback_register,
                        ),
                        DaqNodeSetAction(
                            self.daq,
                            f"{reg_selector_base}/index",
                            awg_config.readout_result_index,
                        ),
                    ]
                )
        return feedback_actions

    def collect_execution_nodes(self, with_pipeliner: bool):
        _logger.debug("Starting execution...")
        nodes = []
        nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/execution/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        nodes.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/triggers/out/0/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        )
        return nodes

    def collect_execution_setup_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeAction]:
        nodes = []
        if with_pipeliner:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/execution/synchronization/enable",
                    1,
                )
            )
        return nodes

    def collect_execution_teardown_nodes(
        self, with_pipeliner: bool
    ) -> list[DaqNodeAction]:
        nodes = []
        if with_pipeliner:
            nodes.append(
                DaqNodeSetAction(
                    self._daq,
                    f"/{self.serial}/execution/synchronization/enable",
                    0,
                )
            )
        return nodes

    def collect_trigger_configuration_nodes(
        self, initialization: Initialization, recipe_data: RecipeData
    ) -> list[DaqNodeAction]:
        # TODO(2K): This was moved as is from no more existing "configure_as_leader".
        # Verify, if separate `batch_set` per node is truly necessary here, or the corresponding
        # nodes can be set in one batch with others.
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
                    initialization.config.reference_clock.value,
                )
            ]
        )

        # Ensure ZSync links are established
        # TODO(2K): This is rather a hotfix, waiting to be done in parallel for all devices with
        # subscription / poll
        # TODO(2K): Verify also the downlink device serial (.../connection/serial) matches
        for port in self._downlinks:
            self._wait_for_node(
                f"/{self.serial}/{port.lower()}/connection/status", 2, timeout=10
            )

        nodes_to_configure_triggers = []

        nodes_to_configure_triggers.append(
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/execution/repetitions",
                initialization.config.repetitions,
            )
        )

        return nodes_to_configure_triggers
