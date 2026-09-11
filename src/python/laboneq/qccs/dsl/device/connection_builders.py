# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The DSL connection builder for QCCS instruments."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device.connection import InternalConnection, SignalConnection
from laboneq.dsl.device.connection_builder import (
    DeviceSetupInternalException,
    ResolvedConnection,
    ResolvedPort,
    physical_channel_type,
)
from laboneq.dsl.enums import IOSignalType

if TYPE_CHECKING:
    from laboneq.dsl.device.instrument import Instrument


def resolve_hdawg_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    _raise_for_invalid_ports(instrument, connection.ports)
    if isinstance(connection, InternalConnection):
        if len(connection.ports) != 1:
            raise DeviceSetupInternalException(
                "Connection to instrument for HDAWG requires exactly one local port."
            )
        return ResolvedConnection(
            ports=[ResolvedPort(connection.ports[0], IOSignalType.DIO)]
        )

    if connection.type not in (None, "iq", "rf"):
        raise DeviceSetupInternalException(
            f"HDAWG connection of unknown type '{connection.type}'."
        )
    if connection.type == "iq" and len(connection.ports) != 2:
        raise DeviceSetupInternalException(
            "HDAWG SIGOUTS connection for requires two local ports, where the "
            "first is the I channel and the second is the Q channel."
        )
    if connection.type == "rf" and len(connection.ports) != 1:
        raise DeviceSetupInternalException(
            "HDAWG connection of type RF requires exactly one port."
        )

    signal_type = "iq" if len(connection.ports) == 2 else "rf"
    ports = []
    if signal_type == "iq":
        if _sigout_channel(connection.ports[0]) > _sigout_channel(connection.ports[1]):
            raise DeviceSetupInternalException(
                f"HDAWG IQ pair {connection.ports} must list the I channel "
                "(the lower port) before the Q channel (the higher port)."
            )
        ports = [
            ResolvedPort(connection.ports[0], IOSignalType.I),
            ResolvedPort(connection.ports[1], IOSignalType.Q),
        ]
    elif signal_type == "rf":
        ports = [ResolvedPort(connection.ports[0], IOSignalType.RF)]
    return ResolvedConnection(
        ports=ports,
        channel_type=physical_channel_type(signal_type),
    )


def resolve_uhfqa_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    _raise_for_invalid_ports(instrument, connection.ports)
    if not isinstance(connection, SignalConnection):
        raise DeviceSetupInternalException(
            "Only signal connections are supported on UHFQA."
        )
    if connection.type not in (None, "iq", "acquire"):
        raise DeviceSetupInternalException(
            f"UHFQA connection of unknown type '{connection.type}'."
        )
    if connection.type == "acquire" and connection.ports:
        raise DeviceSetupInternalException(
            "Specifying ports for UHFQA acquire signal is not allowed."
        )

    signal_type = "acquire" if not connection.ports else "iq"
    channel_type = physical_channel_type(signal_type)
    if signal_type == "acquire":
        return ResolvedConnection(
            ports=[
                ResolvedPort("QAS/0", IOSignalType.IQ),
                ResolvedPort("QAS/1", IOSignalType.IQ),
            ],
            direction=IODirection.IN,
            channel_type=channel_type,
        )
    if len(connection.ports) != 2:
        raise DeviceSetupInternalException(
            "IQ signal connection for UHFQA requires two local ports, where "
            "the first is the I channel and the second is the Q channel."
        )
    if _sigout_channel(connection.ports[0]) > _sigout_channel(connection.ports[1]):
        raise DeviceSetupInternalException(
            f"UHFQA IQ pair {connection.ports} must list the I channel "
            "(the lower port) before the Q channel (the higher port)."
        )
    return ResolvedConnection(
        ports=[
            ResolvedPort(connection.ports[0], IOSignalType.I),
            ResolvedPort(connection.ports[1], IOSignalType.Q),
        ],
        channel_type=channel_type,
    )


def resolve_shfppc_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    _raise_for_invalid_ports(instrument, connection.ports)
    # TODO: Can SHFPPC port be connected to multiple logical signals?
    # Currently no tests for it.
    if not isinstance(connection, SignalConnection):
        raise DeviceSetupInternalException(
            "Only signal connections are supported on SHFPPC."
        )
    if len(connection.ports) != 1:
        raise DeviceSetupInternalException("SHFPPC signals require exactly one port.")
    # A pump channel forms no physical channel of its own.
    return ResolvedConnection(
        ports=[ResolvedPort(connection.ports[0], IOSignalType.PPC)]
    )


def resolve_qa_channel_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    _raise_for_invalid_ports(instrument, connection.ports)
    if not isinstance(connection, SignalConnection):
        raise DeviceSetupInternalException(
            "Only signal connections are supported on QA channels."
        )
    if connection.type not in (None, "acquire", "iq"):
        raise DeviceSetupInternalException(
            "Only acquire and IQ signals are supported on QA channels."
        )
    if len(connection.ports) != 1:
        raise DeviceSetupInternalException(
            "Device QA channel signals require exactly one port."
        )
    port = connection.ports[0]
    signal_type = "acquire" if port.endswith("INPUT") else "iq"
    return ResolvedConnection(
        ports=[ResolvedPort(port, IOSignalType.IQ)],
        direction=IODirection.IN if signal_type == "acquire" else IODirection.OUT,
        channel_type=physical_channel_type(signal_type),
    )


def resolve_sg_channel_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    _raise_for_invalid_ports(instrument, connection.ports)
    if not isinstance(connection, SignalConnection):
        raise DeviceSetupInternalException(
            "Only signal connections are supported on SG channels."
        )
    # Compatibility for DeviceSetup descriptor: It still requires signal
    # type to be defined.
    if connection.type not in (None, "iq"):
        raise DeviceSetupInternalException(
            "Only IQ signals are supported on SG channels."
        )
    if len(connection.ports) != 1:
        raise DeviceSetupInternalException(
            "Device SG channel signals require exactly one port."
        )
    port = connection.ports[0]
    return ResolvedConnection(
        ports=[ResolvedPort(port, IOSignalType.IQ)],
        channel_type=physical_channel_type("iq"),
    )


def resolve_shfqc_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    resolve = (
        resolve_qa_channel_connection
        if any("QACHANNELS" in port for port in connection.ports)
        else resolve_sg_channel_connection
    )
    return resolve(instrument, connection)


def resolve_zsync_connection(
    instrument: Instrument,
    connection: InternalConnection | SignalConnection,
) -> ResolvedConnection:
    _raise_for_invalid_ports(instrument, connection.ports)
    if not isinstance(connection, InternalConnection):
        raise DeviceSetupInternalException(
            "Only to device connections are supported on ZSYNC connections."
        )
    warnings.warn(
        "Explicit ZSYNCS connections are deprecated and currently ignored. "
        "Remove them from the device setup descriptor. This will be a hard "
        "error in a future version.",
        FutureWarning,
        stacklevel=3,
    )
    return ResolvedConnection(
        ports=[ResolvedPort(connection.from_port, IOSignalType.ZSYNC)]
        if connection.from_port is not None
        else []
    )


def _sigout_channel(port: str) -> int:
    """The channel number of a `SIGOUTS/<n>` port."""
    return int(port.rsplit("/", 1)[-1])


def _raise_for_invalid_ports(instrument: Instrument, ports: list[str]) -> None:
    available = [port.uid for port in instrument.ports]
    for port in ports:
        if port not in available:
            raise DeviceSetupInternalException(
                f"Port {port} not defined for the device "
                f"'{type(instrument).__name__}'. Available ports: {available}"
            )
