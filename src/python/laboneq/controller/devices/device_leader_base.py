# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import IntEnum

from laboneq.controller.devices.async_support import ResponseWaiterAsync
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import DeviceBase
from laboneq.controller.devices.node_control import (
    Setting,
    Condition,
    NodeControlBase,
    Response,
    WaitCondition,
)
from laboneq.controller.recipe_processor import RecipeData
from laboneq.controller.util import LabOneQControllerException

_logger = logging.getLogger(__name__)


class ReferenceClockSourceLeader(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1


class DeviceLeaderBase(DeviceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_internal_clock = False

    def update_clock_source(self, force_internal: bool | None):
        self._use_internal_clock = force_internal is True

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        source = (
            ReferenceClockSourceLeader.INTERNAL
            if self._use_internal_clock
            else ReferenceClockSourceLeader.EXTERNAL
        )
        expected_freq = None if self._use_internal_clock else 10e6
        return [
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/freq", expected_freq
            ),
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/sourceactual", source
            ),
            Setting(f"/{self.serial}/system/clocks/referenceclock/in/source", source),
            Response(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
        ]

    def zsync_link_control_nodes(self) -> list[NodeControlBase]:
        nodes = []
        enabled_zsyncs = {}
        for port, down_stream_devices in self._downlinks.items():
            # No command, these nodes will respond to the follower device switching to ZSync
            nodes.append(
                WaitCondition(f"/{self.serial}/{port.lower()}/connection/status", 2),
            )
            for _, dev_ref in down_stream_devices:
                dev = dev_ref()
                if enabled_zsyncs.get(port.lower()) == dev.serial:
                    # Avoid double-enabling the port when it is connected to SHFQC
                    continue
                enabled_zsyncs[port.lower()] = dev.serial
                nodes.append(
                    WaitCondition(
                        f"/{self.serial}/{port.lower()}/connection/serial",
                        dev.serial[3:],
                    ),
                )

        # Todo: Check if no ZSync ports are registered for synchronisation.
        #  If this check fails, then a previous execution may have exited uncleanly.
        #  See discussions in MR !2739.

        return nodes

    async def configure_feedback(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        min_wait_time = recipe_data.recipe.max_step_execution_time
        # This is required because PQSC/QHUB is only receiving the feedback events
        # during the holdoff time, even for a single trigger.
        nc.add("execution/holdoff", min_wait_time)
        enabled_zsyncs = set()
        for port, downstream_devices in self._downlinks.items():
            [p_kind, p_addr] = port.split("/")
            if p_kind != "ZSYNCS":
                continue
            zsync_output = f"zsyncs/{p_addr}/output"
            zsync_base = f"{zsync_output}/registerbank"
            for follower_uid, follower_ref in downstream_devices:
                follower = follower_ref()
                if follower is None:
                    continue
                for awg_key, awg_config in recipe_data.awg_configs.items():
                    if (
                        awg_key.device_uid != follower_uid
                        or awg_config.source_feedback_register in (None, "local")
                    ):
                        continue  # Only consider devices receiving feedback from PQSC/QHUB
                    if p_addr not in enabled_zsyncs:
                        nc.add(f"{zsync_output}/enable", 1)
                        nc.add(f"{zsync_output}/source", 0)
                    enabled_zsyncs.add(p_addr)

                    reg_selector_base = (
                        f"{zsync_base}/sources/{awg_config.fb_reg_target_index}"
                    )
                    nc.add(f"{reg_selector_base}/enable", 1)
                    nc.add(
                        f"{reg_selector_base}/register",
                        awg_config.source_feedback_register,
                    )
                    nc.add(f"{reg_selector_base}/index", awg_config.fb_reg_source_index)
        await self.set_async(nc)

    async def start_execution(self, with_pipeliner: bool):
        _logger.debug("Starting execution...")
        nc = NodeCollector(base=f"/{self.serial}/")

        nc.add("triggers/out/0/enable", 1, cache=False)

        # Select the first ZSYNC port from downlinks.
        first_valid_link_address = next(
            (s for s in self._downlinks.keys() if s.startswith("ZSYNCS")),
            None,
        )
        # Do not touch any setting if there is no valid ZSYNC connection
        if first_valid_link_address is not None:
            _, p_addr_str = first_valid_link_address.split("/")
            p_addr = int(p_addr_str)

            # Why do we set this?
            # Trigger output on PQSC/QHUB mirrors the ZSYNC signals sent through
            # `p_addr`. No valid connection -> no ZSYNC signal -> no trigger output
            # -> (probably nothing on the scope because trigger out is typically connected to a scope)
            nc.add("triggers/out/0/port", p_addr, cache=False)

        nc.add("execution/enable", 1, cache=False)

        await self.set_async(nc)

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, with_pipeliner: bool
    ):
        if with_pipeliner:
            # TODO(2K): Use timeout from connect
            rw = ResponseWaiterAsync(
                api=self._api,
                nodes={
                    f"/{self.serial}/execution/synchronization/enable": 1  # sync enabled
                },
                timeout_s=1.0,
            )
            await rw.prepare()
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add("execution/synchronization/enable", 1)
            await self.set_async(nc)
            if len(await rw.wait()) > 0:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Internal error: Failed to enable synchronization"
                )

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        if with_pipeliner:
            nc.add("execution/synchronization/enable", 0)
        await self.set_async(nc)

    async def configure_trigger(self, recipe_data: RecipeData):
        initialization = recipe_data.get_initialization(self.device_qualifier.uid)
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("system/clocks/referenceclock/out/enable", 1)
        nc.add("execution/repetitions", initialization.config.repetitions)
        await self.set_async(nc)

    async def reset_to_idle(self):
        await super().reset_to_idle()
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("execution/synchronization/enable", 0, cache=False)
        await self.set_async(nc)
