# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List

from laboneq.core.types.enums import ReferenceClockSource
from laboneq.data import EnumReprMixin
from laboneq.data.calibration import Calibration


class IODirection(EnumReprMixin, Enum):
    IN = "in"
    OUT = "out"


class DeviceType(EnumReprMixin, Enum):
    """Zurich Instruments device type.

    `UNMANAGED` is a Zurich Instruments device which is not directly controlled
    by LabOne Q, but which can still be controlled through LabOne Q's interface
    with `zhinst.toolkit`.
    """

    HDAWG = "hdawg"
    PQSC = "pqsc"
    SHFQA = "shfqa"
    SHFSG = "shfsg"
    UHFQA = "uhfqa"
    SHFQC = "shfqc"
    SHFPPC = "shfppc"
    PRETTYPRINTERDEVICE = "prettyprinterdevice"
    UNMANAGED = "unmanaged"


class PhysicalChannelType(EnumReprMixin, Enum):
    IQ_CHANNEL = "iq_channel"
    RF_CHANNEL = "rf_channel"


class PortType(EnumReprMixin, Enum):
    RF = "RF"
    DIO = "DIO"
    ZSYNC = "ZSYNC"


@dataclass(unsafe_hash=True)
class LogicalSignal:
    name: str
    group: str  # Needed for referencing


@dataclass
class PhysicalChannel:
    name: str
    group: str
    type: PhysicalChannelType
    direction: IODirection
    ports: List[Port] = field(default_factory=list)

    def __hash__(self):
        # Todo: Storing `self` in a hashmap is not safe if `self` is mutable!
        #  Instead derive an immutable key (string?) from the object?
        return hash(
            (self.name, self.group, self.type, self.direction, tuple(self.ports))
        )


@dataclass
class ChannelMapEntry:
    """A mapping between physical and logical signal."""

    physical_channel: PhysicalChannel
    logical_signal: LogicalSignal


@dataclass
class ReferenceClock:
    source: ReferenceClockSource | None = (
        None  # `None` means decision is deferred to controller
    )
    frequency: float | None = None


@dataclass(unsafe_hash=True)
class Port:
    """Instrument port."""

    path: str
    type: PortType
    channel: int


@dataclass
class Server:
    """LabOne Dataserver."""

    uid: str
    host: str
    port: int
    leader_uid: str


@dataclass
class Instrument:
    uid: str
    # For ZI devices, the address is the device serial number.
    address: str
    device_type: DeviceType
    server: Server
    device_options: str = None
    interface: str = "1GbE"
    reference_clock: ReferenceClock = field(default_factory=ReferenceClock)
    ports: List[Port] = field(default_factory=list)
    physical_channels: List[PhysicalChannel] = field(default_factory=list)
    connections: List[ChannelMapEntry] = field(default_factory=list)
    device_class: int = 0x0


@dataclass
class LogicalSignalGroup:
    uid: str = None
    logical_signals: Dict[str, LogicalSignal] = field(default_factory=dict)


@dataclass
class SetupInternalConnection:
    """Connections between ports on two devices.

    That is, ports that are connected to each other, rather
    than something to be controlled or measured.
    """

    from_instrument: Instrument
    from_port: Port
    to_instrument: Instrument
    to_port: Port


@dataclass
class QuantumElement:
    uid: str = None
    signals: List[LogicalSignal] = field(default_factory=list)
    parameters: List = field(default_factory=list)


@dataclass
class Qubit(QuantumElement):
    pass


@dataclass
class Setup:
    uid: str
    servers: Dict[str, Server] = field(default_factory=dict)
    instruments: List[Instrument] = field(default_factory=list)
    logical_signal_groups: Dict[str, LogicalSignalGroup] = field(default_factory=dict)
    setup_internal_connections: List[SetupInternalConnection] = field(
        default_factory=list
    )
    calibration: Calibration = field(default_factory=Calibration)
