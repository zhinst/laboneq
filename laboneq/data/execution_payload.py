# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from numpy.typing import ArrayLike

from laboneq.data import EnumReprMixin
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.data.setup_description import ReferenceClockSource

# Added when SHFQC is split into virtual SHFSG & SHFQA
# SHFQC(uid="shfqc") --> SHFSG(uid="shfqc_sg"), SHFQA(uid="shfqc")
VIRTUAL_SHFSG_UID_SUFFIX = "_sg"


#
# Enums
#
class TargetDeviceType(EnumReprMixin, Enum):
    UHFQA = auto()
    HDAWG = auto()
    SHFQA = auto()
    SHFSG = auto()
    SHFPPC = auto()
    PQSC = auto()
    NONQC = auto()
    PRETTYPRINTERDEVICE = auto()


#
# Data Classes
#
@dataclass
class TargetServer:
    uid: str = None
    host: str = None
    port: int = None


class TargetChannelType(EnumReprMixin, Enum):
    UNKNOWN = auto()
    IQ = auto()
    RF = auto()


@dataclass
class TargetChannelCalibration:
    channel_type: TargetChannelType
    ports: list[str]
    voltage_offset: float | None


@dataclass
class TargetDevice:
    uid: str = None
    server: TargetServer = None
    device_serial: str = None
    device_type: TargetDeviceType = None
    device_options: str = None
    interface: str = None
    has_signals: bool | None = None
    connected_outputs: dict[str, list[int]] | None = None
    internal_connections: list[tuple[str, str]] = field(default_factory=list)
    calibrations: list[TargetChannelCalibration] | None = None
    is_qc: bool = False
    qc_with_qa: bool = False
    reference_clock_source: ReferenceClockSource | None = None
    device_class: int = 0x0


@dataclass
class SourceCode:
    uid: str = None
    file_name: str = None  # TODO(2K): This field currently acts as the uid, not requiring a separate file name if the uid is explicit.
    source_text: str = None


@dataclass
class WaveForm:
    uid: str = None
    sampling_rate: float = None
    length_samples: int = None
    samples: ArrayLike = None


@dataclass
class TargetSetup:
    uid: str = None
    servers: list[TargetServer] = field(default_factory=list)
    devices: list[TargetDevice] = field(default_factory=list)


@dataclass
class ExecutionPayload:
    uid: str = None
    target_setup: TargetSetup = None
    compiled_experiment_hash: str = None
    experiment_hash: str = None
    device_setup_hash: str = None
    scheduled_experiment: ScheduledExperiment | None = None
