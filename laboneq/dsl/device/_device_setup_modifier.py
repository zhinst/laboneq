# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides functions to modify existing `DeviceSetup` instance.

The following functions are provided:

    - `add_dataserver()`: Add an dataserver to the device setup
    - `add_instrument()`: Add an instrument to the device setup
    - `add_connection()`: Add an connection the instrument
"""
from __future__ import annotations

import abc
import copy
import itertools
from typing import TYPE_CHECKING

import laboneq.core.path as qct_path
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device.connection import (
    Connection,
    InternalConnection,
    SignalConnection,
)
from laboneq.dsl.device.instrument import Instrument
from laboneq.dsl.device.instruments import (
    HDAWG,
    PQSC,
    SHFPPC,
    SHFQA,
    SHFQC,
    SHFSG,
    UHFQA,
    PRETTYPRINTERDEVICE,
)
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannel,
    PhysicalChannelType,
)
from laboneq.dsl.device.logical_signal_group import LogicalSignal, LogicalSignalGroup
from laboneq.dsl.device.physical_channel_group import PhysicalChannelGroup
from laboneq.dsl.enums import IOSignalType

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup


class DeviceSetupInternalException(LabOneQException):
    pass


class _InstrumentGenerator(abc.ABC):
    @abc.abstractstaticmethod
    def make_connections(
        connection: InternalConnection | SignalConnection,
    ) -> list[Connection]:
        """Make instrument connections."""
        pass

    @abc.abstractstaticmethod
    def make_logical_signal(
        connection: InternalConnection | SignalConnection, pc: PhysicalChannel
    ) -> LogicalSignal:
        """Make logical signal which is associated with the connection."""
        pass

    @classmethod
    def determine_signal_type(cls, ports: list[str]) -> str:
        raise NotImplementedError(
            "Must be implemented if device connections produces physical signals."
        )

    @staticmethod
    def raise_for_invalid_connection(old: Connection, new: Connection):
        equal_iqs = (IOSignalType.I, IOSignalType.Q, IOSignalType.IQ)
        if old.local_port == new.local_port:
            if old.signal_type in equal_iqs and new.signal_type in equal_iqs:
                return
            if old.signal_type != new.signal_type:
                raise DeviceSetupInternalException(
                    f"Multiple signal types specified for {new.local_port}."
                )

    @staticmethod
    def raise_for_existing_connection(old: Connection, new: Connection):
        if old == new:
            raise DeviceSetupInternalException("Connection already exists.")

    @classmethod
    def make_physical_channel(
        cls, instrument: str, ports: list[str], channel_type: str | None = None
    ) -> PhysicalChannel | None:
        split_ports = [port.split(qct_path.Separator) for port in ports]
        signal_name = "_".join(
            (
                group[0]
                for group in itertools.groupby(
                    [x for y in zip(*split_ports) for x in y]
                )
            )
        ).lower()
        path = qct_path.Separator.join(
            [qct_path.PhysicalChannelGroups_Path_Abs, instrument, signal_name]
        )
        if channel_type is None:
            channel_type = cls.determine_signal_type(ports)
        if channel_type in ("iq", "acquire"):
            channel_type_ = PhysicalChannelType.IQ_CHANNEL
        elif channel_type == "rf":
            channel_type_ = PhysicalChannelType.RF_CHANNEL
        else:
            return None
        return PhysicalChannel(
            uid=f"{instrument}/{signal_name}",
            name=signal_name,
            type=channel_type_,
            path=path,
        )


class _HDAWGGenerator(_InstrumentGenerator):
    @staticmethod
    def determine_signal_type(ports: list[str]) -> str:
        if len(ports) == 2:
            return "iq"
        if len(ports) == 1:
            if ports[0].startswith("DIOS"):
                return "acquire"
            else:
                return "rf"

    @staticmethod
    def make_connections(connection: InternalConnection | SignalConnection):
        connections = []
        if isinstance(connection, SignalConnection):
            if connection.type == "iq":
                if len(connection.ports) != 2:
                    raise DeviceSetupInternalException(
                        "HDAWG SIGOUTS connection for requires two local ports, where the first is the I channel and the second is the Q channel."
                    )
            if connection.type == "rf":
                if len(connection.ports) != 1:
                    raise DeviceSetupInternalException(
                        "HDAWG connection of type RF requires exactly one port."
                    )
        connection_type = _HDAWGGenerator.determine_signal_type(connection.ports)
        if isinstance(connection, SignalConnection):
            if connection_type == "iq":
                for i, local_port in enumerate(connection.ports):
                    connections.append(
                        Connection(
                            local_port=local_port,
                            remote_path=qct_path.insert_logical_signal_prefix(
                                connection.uid
                            ),
                            remote_port=str(i),
                            signal_type=[IOSignalType.I, IOSignalType.Q][i],
                        )
                    )
            elif connection_type == "rf":
                conn = Connection(
                    local_port=connection.ports[0],
                    remote_path=qct_path.insert_logical_signal_prefix(connection.uid),
                    remote_port="0",
                    signal_type=IOSignalType.RF,
                )
                connections.append(conn)
        else:
            if len(connection.ports) != 1:
                raise DeviceSetupInternalException(
                    "Connection to instrument for HDAWG requires exactly one local port."
                )
            conn = Connection(
                local_port=connection.ports[0],
                remote_path=connection.to,
                remote_port="0",
                signal_type=IOSignalType.DIO,
            )
            connections.append(conn)
        return connections

    @staticmethod
    def make_logical_signal(connection, pc):
        ls = LogicalSignal(
            uid=connection.uid,
            name=connection.name,
            direction=IODirection.OUT,
            path=qct_path.Separator.join(
                [
                    qct_path.LogicalSignalGroups_Path_Abs,
                    connection.uid,
                ]
            ),
            physical_channel=pc,
        )
        return ls


class _PRETTYPRINTERDEVICEGenerator(_InstrumentGenerator):
    @staticmethod
    def determine_signal_type(ports: list[str]) -> str:
        return "rf"

    @staticmethod
    def make_connections(connection: InternalConnection | SignalConnection):
        connections = []
        if isinstance(connection, SignalConnection):
            if connection.type != "rf" or len(connection.ports) != 1:
                raise DeviceSetupInternalException(
                    "Pretty printer devices are required to have one rf signal on exactly one port."
                )
            conn = Connection(
                local_port=connection.ports[0],
                remote_path=qct_path.insert_logical_signal_prefix(connection.uid),
                remote_port="0",
                signal_type=IOSignalType.RF,
            )
            connections.append(conn)
        else:
            raise DeviceSetupInternalException(
                "Pretty printer devices do not feature a DIO port"
            )
        return connections

    @staticmethod
    def make_logical_signal(connection, pc):
        ls = LogicalSignal(
            uid=connection.uid,
            name=connection.name,
            direction=IODirection.OUT,
            path=qct_path.Separator.join(
                [
                    qct_path.LogicalSignalGroups_Path_Abs,
                    connection.uid,
                ]
            ),
            physical_channel=pc,
        )
        return ls


class _UHFQAGenerator(_InstrumentGenerator):
    @staticmethod
    def determine_signal_type(ports: list[str]) -> str:
        if not ports:
            return "acquire"
        else:
            return "iq"

    @staticmethod
    def make_connections(
        connection: InternalConnection | SignalConnection,
    ) -> list[Connection]:
        if not isinstance(connection, SignalConnection):
            raise DeviceSetupInternalException(
                "Only signal connections are supported on UHFQA."
            )
        if connection.type is not None:
            if connection.type == "rf":
                raise DeviceSetupInternalException("RF signal not supported on UHFQA.")
            if connection.type == "acquire" and connection.ports:
                raise DeviceSetupInternalException(
                    "Specifying ports for UHFQA acquire signal is not allowed."
                )
        connection_type = _UHFQAGenerator.determine_signal_type(connection.ports)
        connections = []
        if connection_type == "acquire":
            for i, local_port in enumerate(["QAS/0", "QAS/1"]):
                connections.append(
                    Connection(
                        local_port=local_port,
                        remote_path=qct_path.insert_logical_signal_prefix(
                            connection.uid
                        ),
                        remote_port=str(i),
                        signal_type=IOSignalType.IQ,
                        direction=IODirection.IN,
                    )
                )
        if connection_type == "iq":
            if len(connection.ports) != 2:
                raise DeviceSetupInternalException(
                    "IQ signal connection for UHFQA requires two local ports, where the first is the I channel and the second is the Q channel."
                )
            for i, local_port in enumerate(connection.ports):
                connections.append(
                    Connection(
                        local_port=local_port,
                        remote_path=qct_path.insert_logical_signal_prefix(
                            connection.uid
                        ),
                        remote_port=str(i),
                        signal_type=[IOSignalType.I, IOSignalType.Q][i],
                    )
                )
        return connections

    @staticmethod
    def make_logical_signal(connection, pc: PhysicalChannel) -> LogicalSignal:
        is_output = True
        if _UHFQAGenerator.determine_signal_type(connection.ports) == "acquire":
            is_output = False
        ls = LogicalSignal(
            uid=connection.uid,
            name=connection.name,
            direction=IODirection.OUT if is_output else IODirection.IN,
            path=qct_path.Separator.join(
                [
                    qct_path.LogicalSignalGroups_Path_Abs,
                    connection.uid,
                ]
            ),
            physical_channel=pc,
        )
        return ls

    @staticmethod
    def make_physical_channel(instrument: str, ports: list[str]):
        connection_type = _UHFQAGenerator.determine_signal_type(ports)
        if connection_type == "acquire":
            ports = ["QAS/0", "QAS/1"]
        return _InstrumentGenerator.make_physical_channel(
            instrument, ports, connection_type
        )


class _SHFPPCGenerator(_InstrumentGenerator):
    @staticmethod
    def make_connections(connection: SignalConnection) -> list[Connection]:
        # TODO: Can SHFPPC port be connected to multiple logical signals? Currently no tests for it.
        connections = []
        if not isinstance(connection, SignalConnection):
            raise DeviceSetupInternalException(
                "Only signal connections are supported on SHFPPC."
            )
        if len(connection.ports) != 1:
            raise DeviceSetupInternalException(
                "SHFPPC signals require exactly one port."
            )
        # SHFPPC accepts either full logical signal path or just <group>/<name>
        remote_path = connection.uid
        if qct_path.LogicalSignalGroups_Path_Abs not in remote_path:
            remote_path = (
                qct_path.LogicalSignalGroups_Path_Abs
                + qct_path.Separator
                + connection.uid
            )
        connections.append(
            Connection(
                local_port=connection.ports[0],
                remote_path=remote_path,
                remote_port=None,
                signal_type=IOSignalType.PPC,
            )
        )
        return connections

    @staticmethod
    def make_logical_signal(connection, pc: PhysicalChannel) -> LogicalSignal:
        return None

    @staticmethod
    def make_physical_channel(instrument: str, ports: list[str]):
        return None


class _SHFQAGenerator(_InstrumentGenerator):
    @staticmethod
    def determine_signal_type(ports: list[str]) -> str:
        if ports[0].endswith("INPUT"):
            return "acquire"
        elif ports[0].endswith("OUTPUT"):
            return "iq"

    @staticmethod
    def make_connections(connection: InternalConnection) -> list[Connection]:
        if not isinstance(connection, SignalConnection):
            raise DeviceSetupInternalException(
                "Only signal connections are supported on QA channels."
            )
        if len(connection.ports) != 1:
            raise DeviceSetupInternalException(
                "Device QA channel signals require exactly one port."
            )
        if connection.type == "rf":
            raise DeviceSetupInternalException(
                "RF signals not supported on QA channels."
            )

        connection_type = _SHFQAGenerator.determine_signal_type(connection.ports)
        connections = []
        if connection_type == "acquire":
            for i, local_port in enumerate(connection.ports):
                connections.append(
                    Connection(
                        local_port=local_port,
                        remote_path=qct_path.insert_logical_signal_prefix(
                            connection.uid
                        ),
                        remote_port=str(i),
                        signal_type=IOSignalType.IQ,
                        direction=IODirection.IN,
                    )
                )
        elif connection_type == "iq":
            for i, local_port in enumerate(connection.ports):
                connections.append(
                    Connection(
                        local_port=local_port,
                        remote_path=qct_path.insert_logical_signal_prefix(
                            connection.uid
                        ),
                        remote_port=str(i),
                        signal_type=IOSignalType.IQ,
                    )
                )
        return connections

    @staticmethod
    def make_logical_signal(connection, pc: PhysicalChannel) -> LogicalSignal:
        is_output = _SHFQAGenerator.determine_signal_type(connection.ports) != "acquire"
        return LogicalSignal(
            uid=connection.uid,
            name=connection.name,
            direction=IODirection.OUT if is_output else IODirection.IN,
            path=qct_path.Separator.join(
                [
                    qct_path.LogicalSignalGroups_Path_Abs,
                    connection.uid,
                ]
            ),
            physical_channel=pc,
        )


class _SHFSGGenerator(_InstrumentGenerator):
    @staticmethod
    def determine_signal_type(ports: list[str]) -> str:
        if ports[0].endswith("OUTPUT"):
            return "iq"

    @staticmethod
    def make_connections(connection: InternalConnection) -> list[Connection]:
        if not isinstance(connection, SignalConnection):
            raise DeviceSetupInternalException(
                "Only signal connections are supported on SG channels."
            )
        # Compatibility for DeviceSetup descriptor: It still requires signal type to be defined.
        if connection.type is not None:
            if not connection.type == "iq":
                raise DeviceSetupInternalException(
                    "Only IQ signals are supported on SG channels."
                )
        if len(connection.ports) != 1:
            raise DeviceSetupInternalException(
                "Device SG channel signals require exactly one port."
            )

        connection_type = _SHFSGGenerator.determine_signal_type(connection.ports)
        connections = []
        if connection_type == "iq":
            for i, local_port in enumerate(connection.ports):
                connections.append(
                    Connection(
                        local_port=local_port,
                        remote_path=qct_path.insert_logical_signal_prefix(
                            connection.uid
                        ),
                        remote_port=str(i),
                        signal_type=IOSignalType.IQ,
                    )
                )
        return connections

    @staticmethod
    def make_logical_signal(connection, pc: PhysicalChannel) -> LogicalSignal:
        return LogicalSignal(
            uid=connection.uid,
            name=connection.name,
            direction=IODirection.OUT,
            path=qct_path.Separator.join(
                [
                    qct_path.LogicalSignalGroups_Path_Abs,
                    connection.uid,
                ]
            ),
            physical_channel=pc,
        )


class _SHFQCGenerator(_InstrumentGenerator):
    @staticmethod
    def determine_signal_type(ports: list[str]) -> str:
        is_qa = False
        for port in ports:
            if "QACHANNELS" in port:
                is_qa = True
                break
        if is_qa:
            return _SHFQAGenerator.determine_signal_type(ports)
        else:
            return _SHFSGGenerator.determine_signal_type(ports)

    @staticmethod
    def make_connections(connection: SignalConnection) -> list[Connection]:
        connections = []
        is_qa = False
        for ports in connection.ports:
            if "QACHANNELS" in ports:
                is_qa = True
                break
        if is_qa:
            qa_conns = _SHFQAGenerator.make_connections(connection)
            connections.extend(qa_conns)
        else:
            sg_conns = _SHFSGGenerator.make_connections(connection)
            connections.extend(sg_conns)
        return connections

    @staticmethod
    def make_logical_signal(connection, pc: PhysicalChannel) -> LogicalSignal:
        is_qa = False
        for ports in connection.ports:
            if "QACHANNELS" in ports:
                is_qa = True
                break
        if is_qa:
            return _SHFQAGenerator.make_logical_signal(connection, pc)
        else:
            return _SHFSGGenerator.make_logical_signal(connection, pc)


class _PQSCGenerator(_InstrumentGenerator):
    @staticmethod
    def make_connections(connection: InternalConnection) -> list[Connection]:
        if not isinstance(connection, InternalConnection):
            raise DeviceSetupInternalException(
                "Only to device connections are supported on PQSC."
            )
        return [
            Connection(
                local_port=connection.from_port,
                remote_path=connection.to,
                remote_port="0",
                signal_type=IOSignalType.ZSYNC,
            )
        ]

    @staticmethod
    def make_logical_signal(connection, pc: PhysicalChannel) -> LogicalSignal:
        pass


def _raise_for_invalid_ports(instr: ZIStandardInstrument, ports: list[str]):
    port_uids = [port.uid for port in instr.ports]
    for port in ports:
        if port not in port_uids:
            msg = f"Port {port} not defined for the device '{instr.__class__.__name__}'. Available ports: {port_uids}"
            raise DeviceSetupInternalException(msg)


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
    HANDLERS = {
        HDAWG: _HDAWGGenerator,
        UHFQA: _UHFQAGenerator,
        SHFPPC: _SHFPPCGenerator,
        SHFQA: _SHFQAGenerator,
        SHFQC: _SHFQCGenerator,
        SHFSG: _SHFSGGenerator,
        PQSC: _PQSCGenerator,
        PRETTYPRINTERDEVICE: _PRETTYPRINTERDEVICEGenerator,
    }
    if dev := setup.instrument_by_uid(instrument):
        handler: _InstrumentGenerator = HANDLERS[dev.__class__]
        _raise_for_invalid_ports(dev, connection.ports)

        # Device connections
        new_conns = handler.make_connections(connection)
        for conn in dev.connections:
            for new_conn in new_conns:
                if isinstance(connection, SignalConnection):
                    _InstrumentGenerator.raise_for_invalid_connection(conn, new_conn)
                    _InstrumentGenerator.raise_for_existing_connection(conn, new_conn)
                if isinstance(connection, InternalConnection):
                    _InstrumentGenerator.raise_for_invalid_connection(conn, new_conn)

        if isinstance(connection, SignalConnection):
            # Physical signals
            pc = handler.make_physical_channel(instrument, connection.ports)
            if pc:
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

                # Logical signals
                assert isinstance(
                    pc, PhysicalChannel
                ), "LogicalSignal must have a physical channel"
                ls = handler.make_logical_signal(connection, pc)
                if ls:
                    group, name = ls.uid.split(qct_path.Separator)
                    if group in setup.logical_signal_groups:
                        if name in setup.logical_signal_groups[group].logical_signals:
                            raise DeviceSetupInternalException(
                                f"Signal {connection.uid} already exists."
                            )
                        else:
                            setup.logical_signal_groups[group].logical_signals[
                                name
                            ] = ls
                    else:
                        lsg = LogicalSignalGroup(uid=group)
                        lsg.logical_signals[name] = ls
                        setup.logical_signal_groups[group] = lsg

        dev.connections.extend(new_conns)
    else:
        raise DeviceSetupInternalException(f"Instrument {instrument} not found.")


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
        if not instrument.address:
            raise DeviceSetupInternalException("Instrument must have an address.")
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
