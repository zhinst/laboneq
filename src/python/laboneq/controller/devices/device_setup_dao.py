# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass, field
import logging
from typing import TYPE_CHECKING, ItemsView, Iterator

from laboneq.controller.versioning import SetupCaps
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
from laboneq.data.execution_payload import (
    TargetChannelCalibration,
    TargetChannelType,
    TargetDeviceType,
)
from laboneq.implementation.utils.devices import parse_device_options

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
    force_internal_clock_source: bool | None = None
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
    # Set the clock source (external by default)
    # TODO(2K): Simplify the logic in this code snippet and the one in 'update_clock_source'.
    # Currently, it adheres to the previously existing logic in the compiler, but it appears
    # unnecessarily convoluted.
    force_internal: bool | None = None
    if target_device.reference_clock_source is not None:
        force_internal = (
            target_device.reference_clock_source == ReferenceClockSource.INTERNAL
        )
    options = DeviceOptions(
        serial=target_device.device_serial,
        interface=target_device.interface,
        dev_type=target_device.device_type.name,
        is_qc=target_device.is_qc,
        qc_with_qa=target_device.qc_with_qa,
        gen2=has_shf,
        force_internal_clock_source=force_internal,
    )

    expected_dev_type, expected_dev_opts = parse_device_options(
        target_device.device_options
    )
    if expected_dev_type is None:
        # TODO(2K): Remove this once all tests provide options explicitly.
        # If no options are given, we set it to the emulator's defaults.
        # This has a side effect that the emulator's defaults would also
        # be used if a user omit options, however it's safe — a later check
        # will fail if the real setup instrument options don't match.
        expected_dev_type = (
            "SHFQC" if target_device.is_qc else target_device.device_type.name
        )
        expected_dev_opts = []
        if expected_dev_type == "HDAWG":
            expected_dev_type = "HDAWG8"
            expected_dev_opts = ["MF", "ME", "SKW", "PC"]
        elif expected_dev_type == "SHFQA":
            expected_dev_type = "SHFQA4"
            expected_dev_opts = ["AWG", "DIG", "QA"]
        elif expected_dev_type == "SHFSG":
            expected_dev_type = "SHFSG8"
        elif expected_dev_type == "SHFQC":
            expected_dev_opts = ["QC6CH"]
        elif expected_dev_type == "SHFPPC":
            expected_dev_type = "SHFPPC4"
    options.expected_dev_type = expected_dev_type
    options.expected_dev_opts = expected_dev_opts
    if (
        not target_device.has_signals
        and not target_device.internal_connections
        and target_device.device_type
        not in (TargetDeviceType.PQSC, TargetDeviceType.QHUB)
    ):
        # Treat devices without defined connections as non-QC,
        # except for PQSC and QHUB which are always known leaders.
        driver = "NONQC"
    if options.is_qc is None:
        options.is_qc = False

    return DeviceQualifier(
        uid=target_device.uid,
        server_uid=target_device.server.uid,
        driver=driver,
        options=options,
    )


def _effective_downlinks(device: TargetDevice, target_setup: TargetSetup) -> list[str]:
    """Return the effective downlinks for a device, excluding SHFPPC and adding implicit PQSC/QHUB links."""
    if device.device_type == TargetDeviceType.SHFPPC:
        return []
    if (
        device.device_type in (TargetDeviceType.PQSC, TargetDeviceType.QHUB)
        and len(device.internal_connections) == 0
    ):
        return [
            dev.uid
            for dev in target_setup.devices
            if dev.device_type
            in (TargetDeviceType.HDAWG, TargetDeviceType.SHFQA, TargetDeviceType.SHFSG)
        ]
    return device.internal_connections


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
        self._downlinks: dict[str, list[str]] = {
            device.uid: _effective_downlinks(device, target_setup)
            for device in target_setup.devices
        }
        self._calibrations: dict[str, list[TargetChannelCalibration]] = {
            device.uid: copy.deepcopy(device.calibrations)
            for device in target_setup.devices
            if device.calibrations is not None
        }

        self._server_device_serials: dict[str, list[str]] = defaultdict(list)
        for device_qualifier in self._devices:
            self._server_device_serials[device_qualifier.server_uid].append(
                device_qualifier.options.serial
            )

        self._setup_caps = setup_caps

    @property
    def servers(self) -> ItemsView[str, ServerQualifier]:
        return self._servers.items()

    @property
    def devices(self) -> Iterator[DeviceQualifier]:
        return iter(self._devices)

    def server_device_serials(self, server_uid: str) -> list[str]:
        return self._server_device_serials[server_uid]

    @property
    def has_uhf(self) -> bool:
        return self._has_uhf

    @property
    def has_qhub(self) -> bool:
        return self._has_qhub

    def downlinks_by_device_uid(self, device_uid: str) -> list[str]:
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
