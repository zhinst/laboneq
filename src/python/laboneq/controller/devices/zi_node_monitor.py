# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from laboneq.controller.devices.device_utils import FloatWithTolerance, is_expected
from laboneq.controller.util import LabOneQControllerException


@dataclass
class Node:
    path: str
    values: list[Any] = field(default_factory=list)
    last: Any | None = None

    def flush(self):
        self.values.clear()

    def peek(self) -> Any | None:
        return None if len(self.values) == 0 else self.values[0]

    def pop(self) -> Any | None:
        return None if len(self.values) == 0 else self.values.pop(0)

    def get_last(self) -> Any | None:
        return self.last

    def append(self, val: dict[str, Any] | list[dict[str, Any]]):
        if isinstance(val, dict):
            # Scalar nodes, value is an array of consecutive updates
            self.values.extend(val["value"])
        else:
            # Vector nodes
            for v in val:
                if isinstance(v["vector"], str):
                    # String value, ignore extra header
                    self.values.append(v["vector"])
                else:
                    # Other vector types, keep everything
                    self.values.append(v)
        self.last = self.values[-1]


class NodeMonitorBase(ABC):
    def __init__(self):
        self._nodes: dict[str, Node] = {}

    @abstractmethod
    async def start(self): ...

    @abstractmethod
    async def stop(self): ...

    @abstractmethod
    async def poll(self): ...

    @abstractmethod
    async def wait_for_state_by_get(self, path: str, expected: int): ...

    def _fail_on_missing_node(self, path: str):
        if path not in self._nodes:
            raise LabOneQControllerException(
                f"Internal error: Node {path} is not registered for monitoring"
            )

    def _get_node(self, path: str) -> Node:
        self._fail_on_missing_node(path)
        return self._nodes[path]

    async def reset(self):
        await self.stop()
        self._nodes.clear()

    def add_nodes(self, paths: list[str]):
        for path in paths:
            if path not in self._nodes:
                self._nodes[path] = Node(path)

    async def flush(self):
        await self.poll()
        for node in self._nodes.values():
            node.flush()

    def peek(self, path: str) -> Any | None:
        return self._get_node(path).peek()

    def pop(self, path: str) -> Any | None:
        return self._get_node(path).pop()

    def get_last(self, path: str) -> Any | None:
        return self._get_node(path).get_last()

    def check_last_for_conditions(
        self, conditions: dict[str, Any]
    ) -> list[tuple[str, Any]]:
        failed: list[tuple[str, Any]] = []
        for path, expected in conditions.items():
            self._fail_on_missing_node(path)
            val = self.get_last(path)
            if not is_expected(val, expected):
                failed.append((path, val))
        return failed

    async def poll_and_check_conditions(
        self, conditions: dict[str, Any]
    ) -> dict[str, Any]:
        await self.poll()
        remaining = {}
        for path, expected in conditions.items():
            self._fail_on_missing_node(path)
            while True:
                val = self.pop(path)
                if val is None:
                    # No further updates for the path,
                    # keep condition as is for the next check iteration
                    remaining[path] = expected
                    break
                if is_expected(val, expected):
                    break
        return remaining


class NodeMonitor(NodeMonitorBase):
    def __init__(self, daq):
        super().__init__()
        self._daq = daq
        self._started = False

    async def start(self):
        if self._started:
            return
        all_paths = [p for p in self._nodes.keys()]
        if len(all_paths) > 0:
            self._daq.subscribe(all_paths)
            for path in all_paths:
                self._daq.getAsEvent(path)
        self._started = True

    async def stop(self):
        self._daq.unsubscribe("*")
        await self.flush()
        self._started = False

    async def poll(self):
        while True:
            data = self._daq.poll(1e-6, 100, flat=True)
            if len(data) == 0:
                break
            for path, val in data.items():
                self._get_node(path).append(val)

    async def wait_for_state_by_get(self, path: str, expected: int):
        if not isinstance(expected, int):
            # Non-int nodes are not important, included only for consistency check.
            # Skip it for this workaround.
            return
        val = next(iter(self._daq.get(path, flat=True).values()))["value"][0]  # deep
        t0 = time.time()
        while time.time() - t0 < 3:  # hard-coded timeout of 3s
            if val == expected:
                return
            await asyncio.sleep(0.005)
            val = self._daq.getInt(path)  # shallow
        raise LabOneQControllerException(
            f"Condition {path}=={expected} is not fulfilled within 3s. Last value: {val}"
        )


class INodeMonitorProvider(ABC):
    @property
    @abstractmethod
    def node_monitor(self) -> NodeMonitorBase: ...


class MultiDeviceHandlerBase:
    def __init__(self):
        self._conditions: dict[NodeMonitorBase, dict[str, Any]] = {}
        self._messages: dict[str, str] = {}

    def add(self, target: INodeMonitorProvider, conditions: dict[str, Any]):
        if conditions:
            daq_conditions: dict[str, Any] = self._conditions.setdefault(
                target.node_monitor, {}
            )
            daq_conditions.update(conditions)

    def add_with_msg(
        self, target: INodeMonitorProvider, conditions: dict[str, tuple[Any, str]]
    ):
        if conditions:
            daq_conditions: dict[str, Any] = self._conditions.setdefault(
                target.node_monitor, {}
            )
            daq_conditions.update({path: val[0] for path, val in conditions.items()})
            self._messages.update({path: val[1] for path, val in conditions.items()})

    def add_from(self, other: MultiDeviceHandlerBase):
        for node_monitor, conditions in other._conditions.items():
            daq_conditions: dict[str, Any] = self._conditions.setdefault(
                node_monitor, {}
            )
            daq_conditions.update(conditions)


class ConditionsChecker(MultiDeviceHandlerBase):
    """Non-blocking checker, ensures all conditions for multiple
    devices are fulfilled. Uses the last known node values, no additional
    polling for updates!

    This class must be prepared in same way as the ResponseWaiter,
    see ResponseWaiter for details.
    """

    def check_all(self) -> list[tuple[str, Any]]:
        failed: list[tuple[str, Any]] = []
        for node_monitor, daq_conditions in self._conditions.items():
            failed.extend(node_monitor.check_last_for_conditions(daq_conditions))
        return failed

    def failed_str(self, failed: list[tuple[str, Any]]) -> str:
        def _find_condition(path: str) -> Any | None | list[Any | None]:
            for daq_conds in self._conditions.values():
                cond = daq_conds.get(path)
                if cond is not None:
                    return cond
            return "<no condition found for the path>"

        return "\n".join(
            [f"{p}: {v}  (expected: {_find_condition(p)})" for p, v in failed]
        )


class ResponseWaiter(MultiDeviceHandlerBase):
    """Parallel waiting for responses from multiple devices over multiple
    connections.

    Usage:
    ======

    apiA = ApiClass('serverA', ...)
    apiB = ApiClass('serverB', ...)

    # One NodeMonitor per data server connection
    monitorA = NodeMonitorNNN(apiA)
    monitorB = NodeMonitorNNN(apiB)

    dev1_monitor = monitorA # dev1 connected via apiA
    dev2_monitor = monitorA # dev2 connected via apiA
    dev3_monitor = monitorB # dev3 connected via apiB
    #...

    dev1_conditions = {
        "/dev1/path1": 5,
        "/dev1/path2": 0,
    }
    dev2_conditions = {
        "/dev2/path1": 3,
        "/dev2/path2": 0,
        # ...
    }
    dev3_conditions = {
        # ...
    }
    #...

    # Register all required conditions with binding to the respective
    # NodeMonitor.
    response_waiter = ResponseWaiter()
    response_waiter.add(target=dev1_monitor, conditions=dev1_conditions)
    response_waiter.add(target=dev2_monitor, conditions=dev2_conditions)
    response_waiter.add(target=dev3_monitor, conditions=dev3_conditions)
    # ...

    # Wait until all the nodes given in the registered conditions return
    # respective values. The call returns 'True' immediately, once all
    # expected responses are received. Times out after 'timeout' seconds (float),
    # returning 'False' in this case.
    if not response_waiter.wait_all(timeout=0.5):
        raise RuntimeError("Expected responses still not received after 2 seconds")
    """

    def __init__(self):
        super().__init__()
        self._timer = time.monotonic

    async def wait_all(self, timeout: float) -> bool:
        start = self._timer()
        while True:
            remaining: dict[NodeMonitorBase, dict[str, Any]] = {}
            for node_monitor, node_monitor_conditions in self._conditions.items():
                node_monitor_remaining = await node_monitor.poll_and_check_conditions(
                    node_monitor_conditions
                )
                if len(node_monitor_remaining) > 0:
                    remaining[node_monitor] = node_monitor_remaining
            if len(remaining) == 0:
                return True
            if self._timer() - start > timeout:
                return False
            self._conditions = remaining

    def remaining(self) -> dict[str, Any]:
        all_conditions: dict[str, Any] = {}
        for daq_conditions in self._conditions.values():
            all_conditions.update(daq_conditions)
        return all_conditions

    def remaining_str(self) -> str:
        failures: list[str] = []
        for p, v in self.remaining().items():
            msg = self._messages.get(p)
            if msg is None:
                msg = f"{p}={v}"
            failures.append(msg)
        return "\n".join(failures)


class NodeControlKind(Enum):
    Condition = auto()
    WaitCondition = auto()
    Setting = auto()
    Command = auto()
    Response = auto()
    Prepare = auto()


@dataclass
class NodeControlBase:
    path: str
    value: Any
    kind: NodeControlKind = None

    @property
    def raw_value(self):
        return (
            self.value.val if isinstance(self.value, FloatWithTolerance) else self.value
        )


@dataclass
class Condition(NodeControlBase):
    """Represents a condition to be fulfilled. Condition node may not
    necessarily receive an update after applying new Setting(s), if it has
    already the right value, for instance extref freq, but it still must be
    verified."""

    def __post_init__(self):
        self.kind = NodeControlKind.Condition


@dataclass
class WaitCondition(NodeControlBase):
    """Represents a condition to be fulfilled. Unlike a plain Condition,
    which causes Setting(s) from the same control block to be applied if not
    fulfilled, the WaitCondition must get fulfilled on its own as a result of
    previously executed control blocks. For example, the ZSync status on PQSC
    is a WaitCondition, which is fulfilled after switching the follower to
    ZSync in a previous action."""

    def __post_init__(self):
        self.kind = NodeControlKind.WaitCondition


@dataclass
class Setting(NodeControlBase):
    """Represents a setting node. The node will be set, if conditions
    of the control block are not fulfilled. Also treated as a response and
    a condition."""

    def __post_init__(self):
        self.kind = NodeControlKind.Setting


@dataclass
class Command(NodeControlBase):
    """Represents a command node. Unlike a setting node, the current value
    of it is not important, but setting this node to a specific value (even
    if it's the same as previously set) triggers a specific activity on the
    instrument, such as loading a preset."""

    def __post_init__(self):
        self.kind = NodeControlKind.Command


@dataclass
class Response(NodeControlBase):
    """Represents a response, expected in return to the changed Setting(s)
    and/or executed Command(s). Also treated as a condition."""

    def __post_init__(self):
        self.kind = NodeControlKind.Response


@dataclass
class Prepare(NodeControlBase):
    """Represents a setting node, that has to be set along with the main
    Setting(s), but shouldn't be touched or be in a specific state otherwise.
    For example, HDAWG outputs must be turned off when changing the system
    clock frequency."""

    def __post_init__(self):
        self.kind = NodeControlKind.Prepare


def _filter_nodes(
    nodes: list[NodeControlBase], filter: list[NodeControlKind]
) -> list[NodeControlBase]:
    return [n for n in nodes if n.kind in filter]


def filter_states(nodes: list[NodeControlBase]) -> list[NodeControlBase]:
    return _filter_nodes(
        nodes,
        [
            NodeControlKind.Condition,
            NodeControlKind.WaitCondition,
            NodeControlKind.Setting,
            NodeControlKind.Response,
        ],
    )


def filter_wait_conditions(nodes: list[NodeControlBase]) -> list[NodeControlBase]:
    return _filter_nodes(nodes, [NodeControlKind.WaitCondition])


def filter_settings(nodes: list[NodeControlBase]) -> list[NodeControlBase]:
    return _filter_nodes(nodes, [NodeControlKind.Prepare, NodeControlKind.Setting])


def filter_responses(nodes: list[NodeControlBase]) -> list[NodeControlBase]:
    return _filter_nodes(nodes, [NodeControlKind.Setting, NodeControlKind.Response])


def filter_commands(nodes: list[NodeControlBase]) -> list[NodeControlBase]:
    return _filter_nodes(nodes, [NodeControlKind.Command])
