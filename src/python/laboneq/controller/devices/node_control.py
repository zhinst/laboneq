# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from laboneq.controller.devices.device_utils import FloatWithTolerance


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
    kind: NodeControlKind | None = None

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
