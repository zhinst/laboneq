# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Conversion of a DSL device setup into the QCCS capnp setup description."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING

from laboneq.qccs.setup import DeviceType
from laboneq.qccs.setup_converter import (
    convert_device_setup_to_setup,
)

if TYPE_CHECKING:
    from laboneq.core.types.enums.io_direction import IODirection
    from laboneq.core.types.enums.physical_channel_type import PhysicalChannelType
    from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.qccs.setup import Setup


@dataclass
class DeviceInfo:
    uid: str
    device_type: DeviceType
    options: str = field(default_factory=str)
    reference_clock_source: ReferenceClockSource | None = None


@dataclass
class PhysicalChannelInfo:
    uid: str
    device_uid: str
    ports: list[str]

    channel_direction: IODirection
    channel_type: PhysicalChannelType


@dataclass
class InternalConnectionInfo:
    from_instrument: str
    from_port: str
    to_instrument: str
    to_port: str


def build(device_setup: DeviceSetup) -> SimpleNamespace:
    """The QCCS setup description for `device_setup`."""
    setup = convert_device_setup_to_setup(device_setup)
    return SimpleNamespace(
        instruments=[_to_instrument(device) for device in _build_devices(setup)],
        signals=[
            SimpleNamespace(
                uid=physical_channel.uid,
                ports=physical_channel.ports,
                instrument_uid=physical_channel.device_uid,
            )
            for physical_channel in _build_physical_channels(setup)
        ],
        internal_connections=_build_internal_connections(setup),
    )


def _to_instrument(device: DeviceInfo) -> SimpleNamespace:
    return SimpleNamespace(
        uid=device.uid,
        device_type=device.device_type.name,
        options=device.options.upper().split("/") if device.options else [],
        reference_clock_source=device.reference_clock_source.name
        if device.reference_clock_source
        else None,
    )


def _build_devices(setup: Setup) -> list[DeviceInfo]:
    """The setup's managed instruments."""
    devices = []
    for instrument in setup.instruments:
        if instrument.device_type == DeviceType.UNMANAGED:
            continue
        device = DeviceInfo(
            uid=instrument.uid,
            device_type=instrument.device_type,
            options=instrument.device_options or "",
            reference_clock_source=instrument.reference_clock.source,
        )
        devices.append(device)
    return devices


def _build_physical_channels(setup: Setup) -> list[PhysicalChannelInfo]:
    """The physical channels of the setup's managed instruments, one per logical signal."""
    signals = []
    for instrument in setup.instruments:
        if instrument.device_type == DeviceType.UNMANAGED:
            continue
        for conn in instrument.connections:
            physical_channel = conn.physical_channel
            logical_signal = conn.logical_signal
            signals.append(
                PhysicalChannelInfo(
                    uid=logical_signal.path(),
                    device_uid=instrument.uid,
                    ports=[p.path for p in physical_channel.ports],
                    channel_direction=physical_channel.direction,
                    channel_type=physical_channel.type,
                )
            )
    return signals


def _build_internal_connections(setup: Setup) -> list[InternalConnectionInfo]:
    """The setup's SHFPPC internal connections."""
    internal_connections = []
    for conn in setup.setup_internal_connections:
        if conn.from_instrument.device_type != DeviceType.SHFPPC:
            continue
        internal_connections.append(
            InternalConnectionInfo(
                from_instrument=conn.from_instrument.uid,
                from_port=conn.from_port.path,
                to_instrument=conn.to_instrument.uid,
                to_port=conn.to_port.path,
            )
        )
    return internal_connections
