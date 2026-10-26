# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A snapshot of a device setup, as the controller derives its devices from it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.core.path import Separator
from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource

if TYPE_CHECKING:
    from laboneq.core.types.enums.io_direction import IODirection
    from laboneq.core.types.enums.physical_channel_type import PhysicalChannelType
    from laboneq.dsl.calibration import Calibration
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.instrument import Instrument
    from laboneq.dsl.device.instruments.zi_standard_instrument import (
        ZIStandardInstrument,
    )
    from laboneq.dsl.device.io_units.logical_signal import LogicalSignal


@dataclass(frozen=True)
class SignalInfo:
    """A logical signal on the channels of one instrument."""

    path: str
    direction: IODirection | None
    channel_type: PhysicalChannelType | None
    voltage_offset: float | None
    #: The ports of the instrument that the signal connects to.
    ports: list[str]


@dataclass(frozen=True)
class InstrumentInfo:
    uid: str
    #: Names the instrument kind, in upper case.
    driver: str
    interface: str
    address: str | None
    device_options: str | None
    #: Empty if the instrument names no server.
    server_uid: str
    #: `True` forces the internal clock source, `False` an external one. `None`
    #: if the instrument specifies none.
    force_internal_clock: bool | None
    has_signals: bool
    signals: list[SignalInfo]
    #: The uids of the instruments this one reaches.
    downlinks: list[str]


@dataclass(frozen=True)
class ServerInfo:
    host: str
    port: int


@dataclass(frozen=True)
class SetupContext:
    """The instruments and servers of a device setup, and whether any instrument is an SHF."""

    instruments: dict[str, InstrumentInfo]
    servers: dict[str, ServerInfo]
    has_shf: bool


def build_setup_context(device_setup: DeviceSetup) -> SetupContext:
    """Takes a snapshot of `device_setup`.

    Raises if an instrument is not a ZI standard instrument, or if its
    connections do not form a valid setup.
    """
    lookups = _Lookups(
        instruments={i.uid: i for i in device_setup.instruments},
        signals={
            cast("str", signal.path): signal
            for group in device_setup.logical_signal_groups.values()
            for signal in group.logical_signals.values()
            if signal.physical_channel is not None
        },
        calibration=device_setup.get_calibration(),
    )
    instruments = {
        i.uid: _instrument_info(i, lookups) for i in device_setup.instruments
    }
    return SetupContext(
        instruments=instruments,
        servers={
            uid: ServerInfo(
                host=server.host,
                # DataServer defines the port as a string.
                port=int(server.port),
            )
            for uid, server in device_setup.servers.items()
        },
        has_shf=any(
            i.driver in ("SHFQA", "SHFSG", "SHFQC") for i in instruments.values()
        ),
    )


@dataclass
class _Lookups:
    instruments: dict[str, Instrument]
    #: The signals bound to a physical channel, by path.
    signals: dict[str, LogicalSignal]
    calibration: Calibration


def _instrument_info(instrument: Instrument, lookups: _Lookups) -> InstrumentInfo:
    # Only a ZI standard instrument has an address, options and a server.
    zi_instrument = cast("ZIStandardInstrument", instrument)
    try:
        address = zi_instrument.address
        device_options = zi_instrument.device_options
        server_uid = zi_instrument.server_uid or ""
        clock_source = zi_instrument.reference_clock_source
    except AttributeError as e:
        raise LabOneQControllerException(
            f"The controller has no handling for instrument "
            f"'{instrument.uid}' of type {type(instrument).__name__}."
        ) from e
    # Set the clock source (external by default)
    # TODO(2K): Simplify the logic in this code snippet and the one in 'update_clock_source'.
    # Currently, it adheres to the previously existing logic in the compiler, but it appears
    # unnecessarily convoluted.
    force_internal = (
        None if clock_source is None else clock_source == ReferenceClockSource.INTERNAL
    )
    return InstrumentInfo(
        uid=instrument.uid,
        driver=instrument.calc_driver().upper(),
        interface=instrument.interface,
        address=address,
        device_options=device_options,
        server_uid=server_uid,
        force_internal_clock=force_internal,
        has_signals=bool(instrument.connections),
        signals=_own_signals(instrument, lookups),
        downlinks=_downlinks(instrument, lookups),
    )


def _own_signals(instrument: Instrument, lookups: _Lookups) -> list[SignalInfo]:
    """Returns each signal on this instrument's own channels with its ports.

    A connection whose port and signal point the same way is this instrument's
    own signal. One pointing the other way reaches another instrument, and is a
    downlink instead.
    """
    ports = {port.uid: port for port in instrument.ports}
    by_path: dict[str, tuple[LogicalSignal, list[str]]] = {}
    for conn in instrument.connections:
        # A connection of a built setup names both of its ends.
        remote_path = cast("str", conn.remote_path)
        local_port = cast("str", conn.local_port)
        signal = lookups.signals.get(remote_path)
        if signal is None or ports[local_port].direction != signal.direction:
            continue
        # A signal reached in the port's own direction must be on a physical
        # channel of this instrument.
        channel_uid = signal.physical_channel.uid
        if _owner_uid(signal, channel_uid) != instrument.uid:
            raise LabOneQControllerException(
                "Instrument to instrument connection ports point to the same"
                f" direction: {instrument.uid}/{local_port} -> {signal.uid}"
                f" -> {channel_uid}"
            )
        _, signal_ports = by_path.setdefault(remote_path, (signal, []))
        signal_ports.append(local_port)
    return [
        _signal_info(signal, signal_ports, lookups.calibration)
        for signal, signal_ports in by_path.values()
    ]


def _signal_info(
    signal: LogicalSignal, ports: list[str], calibration: Calibration
) -> SignalInfo:
    path = cast("str", signal.path)
    signal_calibration = calibration.get(path)
    return SignalInfo(
        path=path,
        direction=signal.direction,
        channel_type=signal.physical_channel.type,
        voltage_offset=(
            None if signal_calibration is None else signal_calibration.voltage_offset
        ),
        ports=ports,
    )


def _downlinks(instrument: Instrument, lookups: _Lookups) -> list[str]:
    """Lists the instruments this one reaches, by uid.

    A connection either names the far instrument outright, or names one of its
    signals, in which case the physical channel's uid names the instrument
    that owns the signal.
    """
    links: list[str] = []
    for conn in instrument.connections:
        remote_path = cast("str", conn.remote_path)
        signal = lookups.signals.get(remote_path)
        if signal is None:
            if conn.signal_type not in (IOSignalType.ZSYNC, IOSignalType.DIO):
                raise LabOneQControllerException(
                    f"No known input port type in device {remote_path} for"
                    f" port type {conn.signal_type}."
                )
            target = lookups.instruments[remote_path]
            # The target must declare a port of the link's own type.
            if not any(port.signal_type == conn.signal_type for port in target.ports):
                raise LabOneQControllerException(
                    f"Instrument '{instrument.uid}' has a connection to an"
                    f" instrument '{target.uid}' without"
                    f" '{conn.signal_type}' ports."
                )
            links.append(remote_path)
        elif signal.direction != conn.direction:
            links.append(_owner_uid(signal, signal.physical_channel.uid))
    return links


def _owner_uid(signal: LogicalSignal, channel_uid: str) -> str:
    """Returns the uid of the instrument owning the channel of `signal`."""
    # A physical channel uid is `<instrument uid>/<channel name>`.
    try:
        return channel_uid.split(Separator)[-2]
    except IndexError as e:
        raise LabOneQControllerException(
            f"Physical channel uid '{channel_uid}' of signal '{signal.uid}'"
            f" must have the form <instrument uid>{Separator}<channel name>."
        ) from e
