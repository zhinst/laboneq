# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, ItemsView, Iterator

from laboneq.controller.versioning import SetupCaps
from laboneq.data.execution_payload import (
    TargetChannelCalibration,
    TargetChannelType,
    TargetDeviceType,
)

if TYPE_CHECKING:
    from laboneq.data.execution_payload import TargetDevice, TargetServer, TargetSetup

_logger = logging.getLogger(__name__)


@dataclass
class DeviceOptions:
    serial: str
    interface: str
    dev_type: str | None = None
    is_qc: bool | None = False
    qc_with_qa: bool = False
    gen2: bool = False
    reference_clock_source: str | None = None
    expected_dev_type: str | None = None
    expected_dev_opts: list[str] = field(default_factory=list)


@dataclass
class ServerQualifier:
    host: str = "localhost"
    port: int = 8004
    ignore_version_mismatch: bool = False


DeviceUID = str


@dataclass
class DeviceQualifier:
    uid: DeviceUID
    server_uid: str
    driver: str
    options: DeviceOptions


def _make_server_qualifier(server: TargetServer, ignore_version_mismatch: bool):
    return ServerQualifier(
        host=server.host,
        port=server.port,
        ignore_version_mismatch=ignore_version_mismatch,
    )


def _make_device_qualifier(
    target_device: TargetDevice, has_shf: bool
) -> DeviceQualifier:
    driver = target_device.device_type.name
    options = DeviceOptions(
        serial=target_device.device_serial,
        interface=target_device.interface,
        dev_type=target_device.device_type.name,
        is_qc=target_device.is_qc,
        qc_with_qa=target_device.qc_with_qa,
        gen2=has_shf,
        reference_clock_source=target_device.reference_clock_source,
    )
    if target_device.device_options is not None:
        opts = target_device.device_options.upper().split("/")
        if len(opts) > 0 and opts[0] == "":
            opts.pop(0)
        if len(opts) > 0:
            options.expected_dev_type = opts.pop(0)
        options.expected_dev_opts = opts
    if not target_device.has_signals and not target_device.internal_connections:
        # Treat devices without defined connections as non-QC
        driver = "NONQC"
        if options.is_qc is None:
            options.is_qc = False

    return DeviceQualifier(
        uid=target_device.uid,
        server_uid=target_device.server.uid,
        driver=driver,
        options=options,
    )


class DeviceSetupDAO:
    # Prevent external deps from spreading throughout the controller.
    @staticmethod
    def is_rf(calib: TargetChannelCalibration) -> bool:
        return calib.channel_type == TargetChannelType.RF

    def __init__(
        self,
        target_setup: TargetSetup,
        setup_caps: SetupCaps,
        ignore_version_mismatch: bool = False,
    ):
        self._target_setup = target_setup
        self._servers: dict[str, ServerQualifier] = {
            server.uid: _make_server_qualifier(
                server=server,
                ignore_version_mismatch=ignore_version_mismatch,
            )
            for server in target_setup.servers
        }

        has_shf = False
        self._has_uhf = False
        self._has_qhub = False
        for device in target_setup.devices:
            if device.device_type in (
                TargetDeviceType.SHFQA,
                TargetDeviceType.SHFSG,
            ):
                has_shf = True
            if device.device_type == TargetDeviceType.UHFQA:
                self._has_uhf = True
            if device.device_type == TargetDeviceType.QHUB:
                self._has_qhub = True

        self._devices: list[DeviceQualifier] = [
            _make_device_qualifier(target_device=device, has_shf=has_shf)
            for device in target_setup.devices
        ]
        self._used_outputs: dict[str, dict[str, list[int]]] = {
            device.uid: device.connected_outputs for device in target_setup.devices
        }
        self._downlinks: dict[str, list[tuple[str, str]]] = {
            device.uid: [i for i in device.internal_connections]
            for device in target_setup.devices
        }
        self._calibrations: dict[str, list[TargetChannelCalibration]] = {
            device.uid: copy.deepcopy(device.calibrations)
            for device in target_setup.devices
            if device.calibrations is not None
        }

        self._setup_caps = setup_caps

    @property
    def servers(self) -> ItemsView[str, ServerQualifier]:
        return self._servers.items()

    @property
    def instruments(self) -> Iterator[DeviceQualifier]:
        return iter(self._devices)

    @property
    def has_uhf(self) -> bool:
        return self._has_uhf

    @property
    def has_qhub(self) -> bool:
        return self._has_qhub

    def downlinks_by_device_uid(self, device_uid: str) -> list[tuple[str, str]]:
        return self._downlinks[device_uid]

    def resolve_ls_path_outputs(self, ls_path: str) -> tuple[str | None, set[int]]:
        for device_uid, used_outputs in self._used_outputs.items():
            outputs = used_outputs.get(ls_path)
            if outputs:
                return device_uid, set(outputs)
        return None, set()

    def get_device_used_outputs(self, device_uid: str) -> set[int]:
        used_outputs: set[int] = set()
        for sig_used_outputs in self._used_outputs[device_uid].values():
            used_outputs.update(sig_used_outputs)
        return used_outputs

    def calibrations(self, device_uid: str) -> list[TargetChannelCalibration]:
        return self._calibrations.get(device_uid) or []

    @property
    def setup_caps(self):
        return self._setup_caps
