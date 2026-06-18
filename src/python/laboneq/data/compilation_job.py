# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.core.types.enums.io_direction import IODirection
    from laboneq.core.types.enums.physical_channel_type import PhysicalChannelType
    from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
    from laboneq.data.calibration import SignalCalibration
    from laboneq.data.setup_description import DeviceType
    from laboneq.data.setup_descriptions import SetupDescription
    from laboneq.dsl.experiment import Experiment


@dataclass
class DeviceInfo:
    uid: str
    device_type: DeviceType
    options: str = field(default_factory=str)
    reference_clock_source: ReferenceClockSource | None = None


@dataclass
class PulseDef:
    uid: str = None
    function: str | None = None
    length: float = None
    amplitude: float = 1.0
    can_compress: bool = False
    samples: ArrayLike | None = None


@dataclass
class ExperimentSignalInfo:
    uid: str
    calibration: SignalCalibration


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


@dataclass
class ExperimentInfo:
    src: Experiment
    signals: list[ExperimentSignalInfo]
    signal_map: dict[str, str]
    devices: list[DeviceInfo]
    physical_channels: list[PhysicalChannelInfo] = field(default_factory=list)
    setup_description: SetupDescription | None = None
    internal_connections: list[InternalConnectionInfo] = field(default_factory=list)
