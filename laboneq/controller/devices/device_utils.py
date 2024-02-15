# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator

import zhinst.core

from laboneq.controller.devices.zi_emulator import EmulatorState

if TYPE_CHECKING:
    from laboneq.controller.devices.device_setup_dao import DeviceSetupDAO
    from laboneq.controller.devices.device_zi import DeviceQualifier


def calc_dev_type(device_qualifier: DeviceQualifier) -> str:
    if device_qualifier.options.is_qc is True:
        return "SHFQC"
    else:
        return device_qualifier.driver


@dataclass
class NodeAction:
    path: str
    value: Any
    cache: bool = True
    filename: str | None = None


class NodeCollector:
    def __init__(self, base: str = ""):
        self._base = base
        self._nodes: list[NodeAction] = []

    def add(
        self, path: str, value: Any, cache: bool = True, filename: str | None = None
    ):
        self._nodes.append(NodeAction(self._base + path, value, cache, filename))

    def extend(self, other: NodeCollector):
        for node in other:
            self._nodes.append(node)

    def __iter__(self) -> Iterator[NodeAction]:
        for node in self._nodes:
            yield node


def zhinst_core_version() -> str:
    [major, minor] = zhinst.core.__version__.split(".")[0:2]
    return f"{major}.{minor}"


def prepare_emulator_state(ds: DeviceSetupDAO) -> EmulatorState:
    emulator_state = EmulatorState()

    # Ensure emulated data server version matches installed zhinst.core
    emulator_state.set_option("ZI", "about/version", zhinst_core_version())

    for device_qualifier in ds.instruments:
        options = device_qualifier.options
        dev_type = calc_dev_type(device_qualifier)
        emulator_state.map_device_type(options.serial, dev_type)
        emulator_state.set_option(options.serial, "dev_type", options.dev_type)
        if options.expected_installed_options is not None:
            exp_opts = options.expected_installed_options.upper().split("/")
            if len(exp_opts) > 0 and exp_opts[0] == "":
                exp_opts.pop(0)
            if len(exp_opts) > 0:
                emulator_state.set_option(
                    options.serial, "features/devtype", exp_opts.pop(0)
                )
            if len(exp_opts) > 0:
                emulator_state.set_option(
                    options.serial, "features/options", "\n".join(exp_opts)
                )

        if dev_type == "PQSC":
            enabled_zsyncs: dict[str, str] = {}
            for from_port, to_dev_uid in ds.downlinks_by_device_uid(
                device_qualifier.uid
            ):
                to_dev_qualifier = next(
                    (i for i in ds.instruments if i.uid == to_dev_uid), None
                )
                if to_dev_qualifier is None:
                    continue
                to_dev_serial = to_dev_qualifier.options.serial.lower()
                if enabled_zsyncs.get(from_port.lower()) == to_dev_serial:
                    continue
                enabled_zsyncs[from_port.lower()] = to_dev_serial
                emulator_state.set_option(
                    options.serial,
                    option=f"{from_port.lower()}/connection/serial",
                    value=to_dev_serial[3:],
                )

    return emulator_state
