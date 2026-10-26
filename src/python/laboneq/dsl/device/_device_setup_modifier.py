# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides functions to modify existing `DeviceSetup` instance.

The following functions are provided:

    - `add_dataserver()`: Add an dataserver to the device setup
    - `add_instrument()`: Add an instrument to the device setup
    - `add_connection()`: Add an connection the instrument
"""

from __future__ import annotations

import copy
import itertools
from typing import TYPE_CHECKING, TypeGuard

import attrs

import laboneq.core.path as qct_path
from laboneq.dsl.device import instrument_registry
from laboneq.dsl.device._calibration_mediator import CalibrationMediator
from laboneq.dsl.device.connection import Connection, SignalConnection
from laboneq.dsl.device.connection_builder import DeviceSetupInternalException
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannel,
)
from laboneq.dsl.device.logical_signal_group import LogicalSignalGroup
from laboneq.dsl.device.physical_channel_group import PhysicalChannelGroup

if TYPE_CHECKING:
    from laboneq.core.types.enums import PhysicalChannelType
    from laboneq.core.types.enums.io_direction import IODirection
    from laboneq.dsl.device.connection import InternalConnection
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.instrument import Instrument


def _raise_for_invalid_connection(old: Connection, new: Connection):
    if old.local_port == new.local_port and old.signal_type != new.signal_type:
        raise DeviceSetupInternalException(
            f"Multiple signal types specified for {new.local_port}."
        )


def _raise_for_existing_connection(old: Connection, new: Connection):
    if old == new:
        raise DeviceSetupInternalException("Connection already exists.")


def _make_physical_channel(
    instrument: str, ports: list[str | None], channel_type: PhysicalChannelType
) -> PhysicalChannel:
    split_ports = [port.split(qct_path.Separator) for port in ports if port is not None]
    signal_name = "_".join(
        (
            group[0]
            for group in itertools.groupby([x for y in zip(*split_ports) for x in y])
        )
    ).lower()
    return PhysicalChannel(
        uid=f"{instrument}/{signal_name}",
        name=signal_name,
        type=channel_type,
        path=qct_path.Separator.join(
            [qct_path.PhysicalChannelGroups_Path_Abs, instrument, signal_name]
        ),
    )


def _make_logical_signal(
    connection: SignalConnection, direction: IODirection, pc: PhysicalChannel
) -> LogicalSignal:
    return LogicalSignal(
        uid=connection.uid,
        name=connection.name,
        direction=direction,
        path=qct_path.Separator.join(
            [qct_path.LogicalSignalGroups_Path_Abs, connection.uid]
        ),
        physical_channel=pc,
    )


def _all_strings(items: list[str | None]) -> TypeGuard[list[str]]:
    return all(x is not None for x in items)


def _resolve_signal_connection_ports(
    setup: DeviceSetup,
    dev: Instrument,
    instrument: str,
    connection: SignalConnection,
) -> list[str]:
    """Resolve a single-entry `ports` spec that names an already-defined
    logical signal on `dev` into the raw port(s) backing that signal.

    If the single entry is already a literal port on `dev`, it is returned
    unchanged. If it matches neither a literal port nor a known logical
    signal, it is also returned unchanged for `ZQCS`, which has no fixed
    port vocabulary and treats it as a brand-new port name. For every other
    instrument type, the function raises.
    """
    [port] = connection.ports
    port_uids = [p.uid for p in dev.ports]
    if port in port_uids:
        return connection.ports

    bare_uid = qct_path.remove_logical_signal_prefix(port)
    ls = None
    group = None
    try:
        group, name = bare_uid.split(qct_path.Separator)
    except ValueError:
        pass
    if group is not None:
        lsg = setup.logical_signal_groups.get(group)
        if lsg is not None:
            ls = lsg.logical_signals.get(name)

    if ls is None or ls.physical_channel is None:
        return connection.ports

    ls_instrument = ls.physical_channel.uid.split(qct_path.Separator)[0]
    if ls_instrument != instrument:
        raise DeviceSetupInternalException(
            f"Logical signal '{port}' cannot be used as a port on "
            f"'{instrument}': it is connected to '{ls_instrument}', not "
            f"'{instrument}'."
        )

    ls_ports = [
        conn.local_port for conn in dev.connections if conn.remote_path == ls.path
    ]
    if _all_strings(ls_ports):
        return ls_ports
    else:
        raise DeviceSetupInternalException(
            f"Logical signal '{port}' cannot be used as a target connection on "
            f"'{instrument}': one of its existing connections is not valid."
        )


def add_connection(
    setup: DeviceSetup,
    instrument: str,
    connection: InternalConnection | SignalConnection,
):
    """Add an connection to an instrument in device setup.

    Modifies given `setup` in place:

        - Updated given instrument connections
        - Updates `DeviceSetup.logical_signal_groups`
        - Updates `DeviceSetup.physical_signal_groups`
    """
    dev = setup.instrument_by_uid(instrument)
    if dev is None:
        raise DeviceSetupInternalException(f"Instrument {instrument} not found.")
    builder = instrument_registry.connection_builder(dev)
    if builder is None:
        raise DeviceSetupInternalException(
            f"Instrument '{instrument}' of type '{type(dev).__name__}' does not "
            f"support connections."
        )

    if isinstance(connection, SignalConnection) and len(connection.ports) == 1:
        connection = attrs.evolve(
            connection,
            ports=_resolve_signal_connection_ports(setup, dev, instrument, connection),
        )

    resolved = builder(dev, connection)

    # Device connections
    remote_path = (
        qct_path.insert_logical_signal_prefix(connection.uid)
        if isinstance(connection, SignalConnection)
        else connection.to
    )
    new_conns = [
        Connection(
            local_port=port.port,
            remote_path=remote_path,
            signal_type=port.signal_type,
            direction=resolved.direction,
        )
        for port in resolved.ports
    ]
    for conn in dev.connections:
        for new_conn in new_conns:
            _raise_for_invalid_connection(conn, new_conn)
            if isinstance(connection, SignalConnection):
                _raise_for_existing_connection(conn, new_conn)

    if isinstance(connection, SignalConnection) and resolved.channel_type is not None:
        # Physical signals
        pc = _make_physical_channel(
            instrument, [port.port for port in resolved.ports], resolved.channel_type
        )
        group, name = pc.uid.split(qct_path.Separator)
        if group in setup.physical_channel_groups:
            if name in setup.physical_channel_groups[group].channels:
                pc = setup.physical_channel_groups[group].channels[name]
            else:
                setup.physical_channel_groups[group].channels[name] = pc
        else:
            pcg = PhysicalChannelGroup(uid=group)
            pcg.channels[name] = pc
            setup.physical_channel_groups[group] = pcg

        # Mediator responsible for making sure calibration changes propagate
        # between the physical channel and the logical signals its mapped to.
        if pc.uid in setup._calibration_mediators:
            mediator = setup._calibration_mediators[pc.uid]
        else:
            mediator = CalibrationMediator(pc)
            setup._calibration_mediators[pc.uid] = mediator

        # Logical signals
        ls = _make_logical_signal(connection, resolved.direction, pc)
        mediator.add_logical_signal(ls)
        group, name = ls.uid.split(qct_path.Separator)
        if group in setup.logical_signal_groups:
            if name in setup.logical_signal_groups[group].logical_signals:
                raise DeviceSetupInternalException(
                    f"Signal {connection.uid} already exists."
                )
            else:
                setup.logical_signal_groups[group].logical_signals[name] = ls
        else:
            lsg = LogicalSignalGroup(uid=group)
            lsg.logical_signals[name] = ls
            setup.logical_signal_groups[group] = lsg

    dev.connections.extend(new_conns)


def add_dataserver(setup: DeviceSetup, dataserver):
    """Add an dataserver to the given device setup.

    Modifies given `setup` in place:

        - Updates `DeviceSetup.servers`
    """
    if dataserver.uid in setup.servers:
        raise DeviceSetupInternalException("Dataserver UIDs must be unique.")
    setup.servers[dataserver.uid] = dataserver


def add_instrument(setup: DeviceSetup, instrument: Instrument):
    """Add an instrument to the given device setup.

    Modifies given `setup` in place:

        - Updates `DeviceSetup.instruments`
    """
    instrument = copy.deepcopy(instrument)
    if not instrument.uid:
        raise DeviceSetupInternalException("Instrument must have an UID.")
    if setup.instrument_by_uid(instrument.uid):
        raise DeviceSetupInternalException(
            "Device setup instrument UIDs must be unique."
        )
    if isinstance(instrument, ZIStandardInstrument):
        if instrument.address in [device.address for device in setup.instruments]:
            raise DeviceSetupInternalException(
                "Device setup instruments must be unique."
            )
        if not instrument.address:
            raise DeviceSetupInternalException("Instrument must have an address.")
        if instrument.requires_data_server:
            if len(setup.servers) == 1 and instrument.server_uid is None:
                instrument.server_uid = next(iter(setup.servers.values())).uid
            else:
                if not setup.servers:
                    raise DeviceSetupInternalException(
                        "At least one dataserver must be defined before instruments."
                    )
                if len(setup.servers) > 1 and instrument.server_uid is None:
                    raise DeviceSetupInternalException(
                        "Multiple dataservers defined. Specify dataserver UID in the instrument `server_uid`."
                    )
                if instrument.server_uid not in setup.servers:
                    raise DeviceSetupInternalException(
                        f"No dataserver '{instrument.server_uid}' defined."
                    )
    setup.instruments.append(instrument)
