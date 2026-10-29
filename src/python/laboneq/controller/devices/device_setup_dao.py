# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ItemsView, Iterator, cast

from laboneq.controller.devices.device_setup_context import build_setup_context
from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.types.enums.physical_channel_type import PhysicalChannelType
from laboneq.core.utilities.device_setup_fingerprint import (
    FingerprintDevice,
    FingerprintDeviceType,
    FingerprintSetup,
)
from laboneq.core.utilities.shfqc import VIRTUAL_SHFSG_UID_SUFFIX, to_sg_uid
from laboneq.implementation.utils.devices import parse_device_options

if TYPE_CHECKING:
    from laboneq.controller.devices.device_setup_context import (
        InstrumentInfo,
        SetupContext,
        SignalInfo,
    )
    from laboneq.controller.versioning import SetupCaps
    from laboneq.dsl.device.device_setup import DeviceSetup

_logger = logging.getLogger(__name__)


@dataclass
class ChannelCalibration:
    """The voltage offset of one channel, and the ports it is applied to."""

    channel_type: PhysicalChannelType | None
    ports: list[str]
    voltage_offset: float | None


@dataclass
class DeviceOptions:
    serial: str
    interface: str
    dev_type: str | None = None
    is_qc: bool | None = False
    qc_with_qa: bool = False
    qc_with_sg: bool = False
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


@dataclass
class DeviceEntry:
    """One device the controller drives, derived from a device setup.

    `device_type` names the kind of instrument. `qualifier.driver` repeats it,
    except `NONQC` for an unused device that is not a leader.
    """

    qualifier: DeviceQualifier
    device_type: str
    connected_outputs: dict[str, list[int]]
    downlinks: list[str]
    calibrations: list[ChannelCalibration]
    device_options: str | None
    is_qc: bool


#: An output port is named `<prefix>/<index>`.
_OUTPUT_PORT_PREFIXES = ("SIGOUTS", "SGCHANNELS", "QACHANNELS")


def _connected_outputs(signals: list[SignalInfo]) -> dict[str, list[int]]:
    outputs: dict[str, list[int]] = {}
    for signal in signals:
        # An acquisition can use a channel whose ports have an output's
        # prefix, so the direction tells the two apart.
        if signal.direction != IODirection.OUT:
            continue
        indices = [
            int(port.split("/")[1])
            for port in signal.ports
            if port.startswith(_OUTPUT_PORT_PREFIXES)
        ]
        if indices:
            outputs.setdefault(signal.path, []).extend(indices)
    return outputs


def _calibrations(signals: list[SignalInfo]) -> list[ChannelCalibration]:
    return [
        ChannelCalibration(
            channel_type=signal.channel_type,
            ports=list(signal.ports),
            voltage_offset=signal.voltage_offset,
        )
        for signal in signals
        if signal.voltage_offset is not None
    ]


def _make_entry(
    instrument: InstrumentInfo,
    device_type: str,
    ctx: SetupContext,
    uid: str,
    connected_outputs: dict[str, list[int]],
    downlinks: list[str],
    calibrations: list[ChannelCalibration],
    has_signals: bool,
    is_qc: bool = False,
    qc_with_qa: bool = False,
    qc_with_sg: bool = False,
) -> DeviceEntry:
    driver = device_type
    options = DeviceOptions(
        serial=cast("str", instrument.address),
        interface=instrument.interface,
        dev_type=device_type,
        is_qc=is_qc,
        qc_with_qa=qc_with_qa,
        qc_with_sg=qc_with_sg,
        gen2=ctx.has_shf,
        force_internal_clock_source=instrument.force_internal_clock,
    )

    expected_dev_type, expected_dev_opts = parse_device_options(
        instrument.device_options
    )
    if expected_dev_type is None:
        # TODO(2K): Remove this once all tests provide options explicitly.
        # If no options are given, we set it to the emulator's defaults.
        # This has a side effect that the emulator's defaults would also
        # be used if a user omit options, however it's safe — a later check
        # will fail if the real setup instrument options don't match.
        expected_dev_type = "SHFQC" if is_qc else device_type
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
    if not has_signals and not downlinks and device_type not in ("PQSC", "QHUB"):
        # Treat devices without defined connections as non-QC,
        # except for PQSC and QHUB which are always known leaders.
        driver = "NONQC"

    return DeviceEntry(
        qualifier=DeviceQualifier(
            uid=uid,
            server_uid=instrument.server_uid,
            driver=driver,
            options=options,
        ),
        device_type=device_type,
        connected_outputs=connected_outputs,
        downlinks=downlinks,
        calibrations=calibrations,
        device_options=instrument.device_options,
        is_qc=is_qc,
    )


def _plain_device_entry(
    instrument: InstrumentInfo, ctx: SetupContext, device_type: str
) -> DeviceEntry:
    return _make_entry(
        instrument,
        device_type,
        ctx,
        uid=instrument.uid,
        connected_outputs=_connected_outputs(instrument.signals),
        downlinks=instrument.downlinks,
        calibrations=_calibrations(instrument.signals),
        has_signals=instrument.has_signals,
    )


def _shfqc_device_entries(
    instrument: InstrumentInfo, ctx: SetupContext
) -> Iterator[DeviceEntry]:
    """Derives the SHFQA and the SHFSG an SHFQC is driven as.

    The SG half is always present, the QA half only if the descriptor gives the
    instrument QA channels. Neither has downlinks, because an SHFQC has no
    ZSync output and cannot lead a follower.
    """
    signals = instrument.signals
    qa_signals = [s for s in signals if not any("SGCHANNELS" in p for p in s.ports)]
    sg_signals = [s for s in signals if not any("QACHANNELS" in p for p in s.ports)]
    sg_outputs = _connected_outputs(sg_signals)
    has_signals = instrument.has_signals

    if qa_signals:
        yield _make_entry(
            instrument,
            "SHFQA",
            ctx,
            uid=instrument.uid,
            connected_outputs=_connected_outputs(qa_signals),
            downlinks=[],
            calibrations=_calibrations(qa_signals),
            has_signals=has_signals,
            is_qc=True,
            qc_with_qa=True,
            qc_with_sg=sg_outputs != {},
        )

    yield _make_entry(
        instrument,
        "SHFSG",
        ctx,
        uid=to_sg_uid(instrument.uid),
        connected_outputs=sg_outputs,
        downlinks=[],
        calibrations=_calibrations(sg_signals),
        has_signals=has_signals,
        is_qc=True,
        qc_with_qa=bool(qa_signals),
        qc_with_sg=sg_outputs != {},
    )


def _effective_downlinks(entry: DeviceEntry, entries: list[DeviceEntry]) -> list[str]:
    """Returns the links this device drives, including those the descriptor omits.

    A PPC connects to an SHFQA over a pump line and leads no follower, so it
    has no links. A leader that names no links drives every follower in the
    setup that has signals, and a named link to an SHFQC reaches both of the
    devices it is driven as.
    """
    if entry.device_type == "SHFPPC":
        return []
    if entry.device_type not in ("PQSC", "QHUB"):
        return entry.downlinks
    if not entry.downlinks:
        return [
            e.qualifier.uid
            for e in entries
            if e.device_type in ("HDAWG", "SHFQA", "SHFSG")
            and e.qualifier.driver != "NONQC"
        ]
    split_qcs = {
        e.qualifier.uid.removesuffix(VIRTUAL_SHFSG_UID_SUFFIX)
        for e in entries
        if e.is_qc
    }
    return entry.downlinks + [
        to_sg_uid(uid) for uid in entry.downlinks if uid in split_qcs
    ]


def build_server_qualifiers(
    ctx: SetupContext, ignore_version_mismatch: bool
) -> dict[str, ServerQualifier]:
    """Builds a qualifier for every server the setup defines.

    Includes a server that no instrument names.
    """
    return {
        uid: ServerQualifier(
            host=server.host,
            port=server.port,
            ignore_version_mismatch=ignore_version_mismatch,
        )
        for uid, server in ctx.servers.items()
    }


def build_device_entries(ctx: SetupContext) -> list[DeviceEntry]:
    """Builds one entry per device the controller drives for the setup.

    One instrument is not always one device. An SHFQC is driven as an SHFQA
    and an SHFSG, each with its own uid.
    """
    entries: list[DeviceEntry] = []
    for instrument in ctx.instruments.values():
        if instrument.driver == "SHFQC":
            entries.extend(_shfqc_device_entries(instrument, ctx))
            continue
        entries.append(_plain_device_entry(instrument, ctx, instrument.driver))
    for entry in entries:
        entry.downlinks = _effective_downlinks(entry, entries)
    return entries


class DeviceSetupDAO:
    # Prevent external deps from spreading throughout the controller.
    @staticmethod
    def is_rf(calib: ChannelCalibration) -> bool:
        return calib.channel_type == PhysicalChannelType.RF_CHANNEL

    def __init__(
        self,
        device_setup: DeviceSetup,
        setup_caps: SetupCaps,
        ignore_version_mismatch: bool = False,
    ):
        ctx = build_setup_context(device_setup)
        entries = build_device_entries(ctx)
        self._servers = build_server_qualifiers(ctx, ignore_version_mismatch)
        self._devices = [entry.qualifier for entry in entries]
        self._has_uhf = any(e.device_type == "UHFQA" for e in entries)
        self._has_qhub = any(e.device_type == "QHUB" for e in entries)
        self._used_outputs = {e.qualifier.uid: e.connected_outputs for e in entries}
        self._downlinks = {e.qualifier.uid: e.downlinks for e in entries}
        self._calibrations = {
            e.qualifier.uid: e.calibrations for e in entries if e.calibrations
        }
        # The `NONQC` driver drives a device with no signals. Its fingerprint
        # still names the instrument kind, which `device_type` keeps.
        self._fingerprint_setup = FingerprintSetup(
            devices=[
                FingerprintDevice(
                    uid=e.qualifier.uid,
                    device_type=FingerprintDeviceType(e.device_type),
                    device_options=e.device_options,
                    is_qc=e.is_qc,
                )
                for e in entries
            ]
        )
        self._server_device_serials = defaultdict(list)
        for qualifier in self._devices:
            self._server_device_serials[qualifier.server_uid].append(
                qualifier.options.serial
            )
        self._setup_caps = setup_caps

    @property
    def fingerprint_setup(self) -> FingerprintSetup:
        return self._fingerprint_setup

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

    def calibrations(self, device_uid: str) -> list[ChannelCalibration]:
        return self._calibrations.get(device_uid) or []

    @property
    def setup_caps(self):
        return self._setup_caps
