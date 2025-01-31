# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from weakref import ReferenceType

from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import DeviceBase


class AwgPipeliner:
    def __init__(
        self, parent_device: ReferenceType[DeviceBase], node_base: str, unit: str
    ):
        self._parent_device = parent_device
        self._node_base = node_base
        self._unit = unit
        self._pipeliner_slot_tracker: dict[int, int] = {}

    @property
    def parent_device(self) -> DeviceBase:
        parent_device = self._parent_device()
        assert parent_device is not None
        return parent_device

    def prepare_for_upload(self, index: int) -> NodeCollector:
        self._pipeliner_slot_tracker[index] = 0
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add(f"{index}/pipeliner/mode", 1)
        nc.add(f"{index}/pipeliner/reset", 1, cache=False)
        nc.add(f"{index}/synchronization/enable", 1)
        nc.barrier()
        return nc

    def commit(self, index: int) -> NodeCollector:
        self._pipeliner_slot_tracker[index] += 1
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.barrier()
        nc.add(f"{index}/pipeliner/commit", 1, cache=False)
        return nc

    def ready_conditions(self, index: int) -> dict[str, Any]:
        max_slots = 1024  # TODO(2K): read on connect from pipeliner/maxslots
        avail_slots = max_slots - self._pipeliner_slot_tracker[index]
        return {f"{self._node_base}/{index}/pipeliner/availableslots": avail_slots}

    def collect_execution_nodes(self) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        for index in self.parent_device._allocated_awgs:
            nc.add(f"{index}/pipeliner/enable", 1, cache=False)
        return nc

    def conditions_for_execution_ready(self) -> dict[str, tuple[Any, str]]:
        if self.parent_device.is_standalone():
            return {
                f"{self._node_base}/{index}/pipeliner/status": (
                    [1, 0],  # exec -> idle
                    f"{self.parent_device.dev_repr}: Pipeliner for {self._unit} channel {index + 1} failed to transition to exec and back to stop.",
                )
                for index in self.parent_device._allocated_awgs
            }
        return {
            f"{self._node_base}/{index}/pipeliner/status": (
                1,  # exec
                f"{self.parent_device.dev_repr}: Pipeliner for {self._unit} channel {index + 1} didn't start.",
            )
            for index in self.parent_device._allocated_awgs
        }

    def conditions_for_execution_done(self) -> dict[str, tuple[Any, str]]:
        if self.parent_device.is_standalone():
            return {}
        return {
            f"{self._node_base}/{index}/pipeliner/status": (
                0,  # idle
                f"{self.parent_device.dev_repr}: Pipeliner for {self._unit} channel {index + 1} didn't stop. Missing start trigger? Check HW synchronization participants.",
            )
            for index in self.parent_device._allocated_awgs
        }

    def reset_nodes(self) -> NodeCollector:
        nc = NodeCollector(base=f"{self._node_base}/")
        nc.add("*/pipeliner/mode", 0, cache=False)  # off
        nc.add("*/synchronization/enable", 0, cache=False)
        return nc
