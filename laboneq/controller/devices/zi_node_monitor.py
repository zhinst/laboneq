# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    path: str
    values: List[Any] = field(default_factory=list)
    last: Optional[Any] = None

    def flush(self):
        self.values.clear()

    def peek(self) -> Optional[Any]:
        return None if len(self.values) == 0 else self.values[0]

    def pop(self) -> Optional[Any]:
        return None if len(self.values) == 0 else self.values.pop(0)

    def append(self, val: Dict[str, Any]):
        self.values.extend(val["value"])
        self.last = self.values[-1]


class NodeMonitor:
    def __init__(self, daq):
        self._daq = daq
        self._logger = logging.getLogger(__name__)
        self._nodes: Dict[str, Node] = {}

    def _log_missing_node(self, path: str):
        self._logger.warning(
            "Internal error: Node %s is not registered for monitoring", path
        )

    def _get_node(self, path: str) -> Node:
        node = self._nodes.get(path)
        if node is None:
            self._log_missing_node(path)
            return Node(path)
        return node

    def reset(self):
        self.stop()
        self._nodes.clear()

    def add_nodes(self, paths: List[str]):
        for path in paths:
            if path not in self._nodes:
                self._nodes[path] = Node(path)

    def start(self):
        all_paths = [p for p in self._nodes.keys()]
        self._daq.subscribe(all_paths)
        for path in all_paths:
            self._daq.getAsEvent(path)

    def stop(self):
        all_paths = [p for p in self._nodes.keys()]
        self._daq.unsubscribe(all_paths)
        self.flush()

    def poll(self):
        while True:
            data = self._daq.poll(1e-6, 100, flat=True)
            if len(data) == 0:
                break
            for path, val in data.items():
                self._get_node(path).append(val)

    def flush(self):
        self.poll()
        for node in self._nodes.values():
            node.flush()

    def peek(self, path: str) -> Optional[Any]:
        return self._get_node(path).peek()

    def pop(self, path: str) -> Optional[Any]:
        return self._get_node(path).pop()

    def check_for_conditions(self, conditions: Dict[str, Any]) -> Dict[str, Any]:
        self.poll()
        for path in conditions.keys():
            if path not in self._nodes:
                self._log_missing_node(path)
                return conditions
        remaining = {}
        for path, expected in conditions.items():
            while True:
                val = self.pop(path)
                if val is None:
                    remaining[path] = expected
                    break
                if val == expected:
                    break
        return remaining


class AllConditionsWaiter:
    def __init__(self):
        self._conditions: Dict[NodeMonitor, Dict[str, Any]] = {}
        self._timer = time.time

    def add(self, target: NodeMonitor, conditions: Dict[str, Any]):
        daq_conditions: Dict[str, Any] = self._conditions.setdefault(target, {})
        daq_conditions.update(conditions)

    def wait_all(self, timeout: float) -> bool:
        start = self._timer()
        while True:
            remaining: Dict[NodeMonitor, Dict[str, Any]] = {}
            for node_monitor, daq_conditions in self._conditions.items():
                daq_remaining = node_monitor.check_for_conditions(daq_conditions)
                if len(daq_remaining) > 0:
                    remaining[node_monitor] = daq_remaining
            if len(remaining) == 0:
                return True
            if self._timer() - start > timeout:
                return False
            self._conditions = remaining
