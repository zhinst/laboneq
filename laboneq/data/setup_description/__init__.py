# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List

from laboneq.data import EnumReprMixin
from laboneq.data.calibration import Calibration


#
# Enums
#
class IODirection(EnumReprMixin, Enum):
    IN = auto()
    OUT = auto()


class ReferenceClockSource(EnumReprMixin, Enum):
    EXTERNAL = auto()
    INTERNAL = auto()


class DeviceType(EnumReprMixin, Enum):
    HDAWG = auto()
    NonQC = auto()
    PQSC = auto()
    SHFQA = auto()
    SHFSG = auto()
    UHFQA = auto()
    SHFQC = auto()


class PhysicalChannelType(EnumReprMixin, Enum):
    IQ_CHANNEL = auto()
    RF_CHANNEL = auto()


class PortType(EnumReprMixin, Enum):
    RF = "RF"
    DIO = "DIO"
    ZSYNC = "ZSYNC"


#
# Data Classes
#


@dataclass
class LogicalSignal:
    name: str
    group: str  # Needed for referencing. TODO(MH): Remove


@dataclass
class PhysicalChannel:
    name: str
    type: PhysicalChannelType = None
    direction: IODirection = None
    ports: List[Port] = None


@dataclass
class ChannelMapEntry:
    """A mapping between physical and logical signal."""

    physical_channel: PhysicalChannel
    logical_signal: LogicalSignal


@dataclass
class ReferenceClock:
    source: ReferenceClockSource = ReferenceClockSource.EXTERNAL
    frequency: float | None = None


@dataclass
class Port:
    path: str
    type: PortType


@dataclass
class Server:
    uid: str = None
    api_level: int = None
    host: str = None
    leader_uid: str = None
    port: int = None


@dataclass
class Instrument:
    uid: str = None
    interface: str = None
    reference_clock: ReferenceClock = ReferenceClock()
    ports: List[Port] = field(default_factory=list)
    physical_channels: List[PhysicalChannel] = field(default_factory=list)
    connections: List[ChannelMapEntry] = field(default_factory=list)
    address: str = None
    device_type: DeviceType = None
    server: Server = None


@dataclass
class LogicalSignalGroup:
    uid: str = None
    logical_signals: Dict[str, LogicalSignal] = field(default_factory=dict)


@dataclass
class SetupInternalConnection:
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
    uid: str = None
    servers: Dict[str, Server] = field(default_factory=dict)
    instruments: List[Instrument] = field(default_factory=list)
    logical_signal_groups: Dict[str, LogicalSignalGroup] = field(default_factory=dict)
    setup_internal_connections: List[SetupInternalConnection] = field(
        default_factory=list
    )
    calibration: Calibration = None
