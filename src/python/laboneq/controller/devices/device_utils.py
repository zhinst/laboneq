# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections.abc import Iterable, Iterator
import math

from dataclasses import dataclass
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

from laboneq.controller.devices.zi_emulator import EmulatorState
from laboneq.controller.versioning import LabOneVersion

if TYPE_CHECKING:
    from laboneq.controller.devices.device_setup_dao import DeviceSetupDAO
    from laboneq.controller.devices.device_zi import DeviceQualifier


def calc_dev_type(device_qualifier: DeviceQualifier) -> str:
    if device_qualifier.options.is_qc is True:
        return "SHFQC"
    else:
        return device_qualifier.driver


@dataclass
class FloatWithTolerance:
    val: float
    abs_tol: float


def is_expected(val: Any, expected: Any | None | list[Any | None]) -> bool:
    if val is None:
        return False

    all_expected = (
        expected
        if isinstance(expected, Iterable) and not isinstance(expected, str)
        else [expected]
    )

    for e in all_expected:
        if e is None:
            # No specific value expected, any update matches
            return True
        if isinstance(e, FloatWithTolerance) and math.isclose(
            val, e.val, abs_tol=e.abs_tol
        ):
            # Float with given tolerance
            return True
        if val == e:
            # Otherwise exact match
            return True
    return False


@dataclass
class NodeAction:
    pass


@dataclass
class NodePath(NodeAction):
    """Use to collect node paths only, e.g. for subscribe."""

    path: str


@dataclass
class NodeActionSet(NodeAction):
    path: str
    value: Any
    cache: bool = True
    filename: str | None = None


@dataclass
class NodeActionBarrier(NodeAction):
    pass


class NodeCollector:
    def __init__(self, base: str = ""):
        self._base = base
        self._nodes: list[NodeAction] = []

    def add(
        self, path: str, value: Any, cache: bool = True, filename: str | None = None
    ):
        self._nodes.append(NodeActionSet(self._base + path, value, cache, filename))

    def add_path(self, path: str):
        self._nodes.append(NodePath(self._base + path))

    def add_node_action(self, node_action: NodeAction):
        self._nodes.append(node_action)

    def barrier(self):
        self._nodes.append(NodeActionBarrier())

    def extend(self, other: Iterable[NodeAction]):
        for node in other:
            self._nodes.append(node)

    def __iter__(self) -> Iterator[NodeAction]:
        for node in self._nodes:
            yield node

    def set_actions(self) -> Iterator[NodeActionSet]:
        for node in self._nodes:
            if isinstance(node, NodeActionSet):
                yield node

    def paths(self) -> Iterator[str]:
        for node in self._nodes:
            if isinstance(node, NodePath):
                yield node.path

    @staticmethod
    def one(
        path: str, value: Any, cache: bool = True, filename: str | None = None
    ) -> NodeCollector:
        nc = NodeCollector()
        nc.add(path=path, value=value, cache=cache, filename=filename)
        return nc

    @staticmethod
    def all(node_collectors: NodeCollector | Iterable[NodeCollector]) -> NodeCollector:
        if isinstance(node_collectors, NodeCollector):
            return node_collectors
        all_nodes = NodeCollector()
        for nc in node_collectors:
            all_nodes.extend(nc)
        return all_nodes


def zhinst_core_version() -> str:
    return version("zhinst-core")


def prepare_emulator_state(ds: DeviceSetupDAO) -> EmulatorState:
    emulator_state = EmulatorState()

    # Ensure emulated data server version matches installed zhinst.core
    #
    labonever = LabOneVersion.from_version_string(zhinst_core_version())
    emulator_state.set_option("ZI", "about/fullversion", str(labonever))

    for device_qualifier in ds.devices:
        options = device_qualifier.options
        dev_type = calc_dev_type(device_qualifier)
        emulator_state.map_device_type(options.serial, dev_type)
        emulator_state.set_option(options.serial, "dev_type", options.dev_type)
        if options.expected_dev_type is not None:
            emulator_state.set_option(
                options.serial, "features/devtype", options.expected_dev_type
            )
            emulator_state.set_option(
                options.serial, "features/options", "\n".join(options.expected_dev_opts)
            )

        if dev_type in ["PQSC", "QHUB"]:
            assigned_zsyncs: set[str] = set()
            from_port = 0
            for to_dev_uid in ds.downlinks_by_device_uid(device_qualifier.uid):
                to_dev_qualifier = next(
                    (i for i in ds.devices if i.uid == to_dev_uid), None
                )
                if to_dev_qualifier is None:
                    continue
                to_dev_serial = to_dev_qualifier.options.serial.lower()
                if to_dev_serial in assigned_zsyncs:
                    continue
                assigned_zsyncs.add(to_dev_serial)
                emulator_state.set_option(
                    options.serial,
                    option=f"zsyncs/{from_port}/connection/serial",
                    value=to_dev_serial[3:],
                )
                from_port += 1

    return emulator_state


def to_l1_timeout(timeout_s: float) -> int:
    assert 0 < timeout_s < float("inf")
    return int(timeout_s * 1e3)
