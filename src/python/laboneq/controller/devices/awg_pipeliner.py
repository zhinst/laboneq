# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from laboneq.controller.devices.device_utils import NodeCollector


class AwgPipeliner:
    def __init__(self, node_base: str, unit: str):
        self._node_base = node_base
        self._unit = unit
        self._pipeliner_slot_tracker = 0

    def prepare_for_upload(self) -> NodeCollector:
        self._pipeliner_slot_tracker = 0
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add("pipeliner/mode", 1)
        nc.add("pipeliner/reset", 1, cache=False)
        nc.add("synchronization/enable", 1)
        nc.barrier()
        return nc

    def commit(self) -> NodeCollector:
        self._pipeliner_slot_tracker += 1
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.barrier()
        nc.add("pipeliner/commit", 1, cache=False)
        return nc

    def ready_conditions(self) -> dict[str, Any]:
        max_slots = 1024  # TODO(2K): read on connect from pipeliner/maxslots
        avail_slots = max_slots - self._pipeliner_slot_tracker
        return {f"{self._node_base}/pipeliner/availableslots": avail_slots}

    def collect_execution_nodes(self) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add("pipeliner/enable", 1, cache=False)
        return nc

    def conditions_for_execution_ready(self) -> dict[str, tuple[Any, str]]:
        return {
            f"{self._node_base}/pipeliner/status": (
                1,  # exec
                f"Pipeliner for {self._unit} didn't start.",
            )
        }

    def conditions_for_execution_done(
        self, with_execution_start: bool = False
    ) -> dict[str, tuple[Any, str]]:
        """
        If with_execution_start is True, constructs a condition that checks for execution start,
        before the actual execution done condition. This is needed since in some situations (e.g. standalone HDAWG),
        the execution happens in one go, and checking for these two states cannot be separated.
        """

        if with_execution_start:
            return {
                f"{self._node_base}/pipeliner/status": (
                    [1, 0],  # exec -> idle
                    f"Pipeliner for {self._unit} failed to transition to exec and back to stop.",
                )
            }
        return {
            f"{self._node_base}/pipeliner/status": (
                0,  # idle
                f"Pipeliner for {self._unit} didn't stop. Missing start trigger? Check HW synchronization participants.",
            )
        }

    def reset_nodes(self) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add("pipeliner/mode", 0, cache=False)  # off
        nc.add("synchronization/enable", 0, cache=False)
        return nc
