# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeAction,
    DaqNodeSetAction,
)


class _MixInToDevice(Protocol):
    _daq: Any
    _allocated_awgs: set[int]

    def _get_num_awgs(self) -> int:
        ...


if TYPE_CHECKING:
    _type_base = _MixInToDevice
else:
    _type_base = object


class AwgPipeliner(_type_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._node_base = ""
        self._pipeliner_slot_tracker: list[int] = []

    @property
    def has_pipeliner(self) -> bool:
        return True

    def pipeliner_set_node_base(self, node_base: str):
        self._node_base = node_base

    def pipeliner_control_nodes(self, index: int) -> list[str]:
        return [
            f"{self._node_base}/{index}/pipeliner/availableslots",
            f"{self._node_base}/{index}/pipeliner/status",
        ]

    def pipeliner_prepare_for_upload(self, index: int) -> list[DaqNodeAction]:
        self._pipeliner_slot_tracker = [0] * self._get_num_awgs()
        return [
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/{index}/pipeliner/mode",
                1,
            ),
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/{index}/pipeliner/reset",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/{index}/synchronization/enable",
                1,
            ),
        ]

    def pipeliner_commit(self, index: int) -> list[DaqNodeAction]:
        self._pipeliner_slot_tracker[index] += 1
        return [
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/{index}/pipeliner/commit",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
        ]

    def pipeliner_ready_conditions(self, index: int) -> dict[str, Any]:
        max_slots = 1024  # TODO(2K): read on connect from pipeliner/maxslots
        avail_slots = max_slots - self._pipeliner_slot_tracker[index]
        return {f"{self._node_base}/{index}/pipeliner/availableslots": avail_slots}

    def pipeliner_collect_execution_nodes(self) -> list[DaqNodeAction]:
        return [
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/{index}/pipeliner/enable",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
            for index in self._allocated_awgs
        ]

    def pipeliner_conditions_for_execution_ready(self) -> dict[str, Any]:
        return {
            f"{self._node_base}/{index}/pipeliner/status": 1  # exec
            for index in self._allocated_awgs
        }

    def pipeliner_conditions_for_execution_done(self) -> dict[str, Any]:
        return {
            f"{self._node_base}/{index}/pipeliner/status": 0  # idle
            for index in self._allocated_awgs
        }

    def pipeliner_reset_nodes(self) -> list[DaqNodeAction]:
        return [
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/*/pipeliner/mode",
                0,  # off
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
            DaqNodeSetAction(
                self._daq,
                f"{self._node_base}/*/synchronization/enable",
                0,
                caching_strategy=CachingStrategy.NO_CACHE,
            ),
        ]
