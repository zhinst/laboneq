# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from typing_extensions import TypeAlias

ParameterUID: TypeAlias = str


class SignalType(Enum):
    IQ = "iq"
    SINGLE = "single"
    INTEGRATION = "integration"
    MARKER = "marker"


class RefClkType(Enum):
    _10MHZ = 10_000_000
    _100MHZ = 100_000_000


class TriggeringMode(Enum):
    ZSYNC_FOLLOWER = 1
    DIO_FOLLOWER = 2
    DESKTOP_LEADER = 3
    DESKTOP_DIO_FOLLOWER = 4
    INTERNAL_FOLLOWER = 5


@dataclass(frozen=True)
class NtStepKey:
    indices: tuple[int, ...]

    def __post_init__(self):
        # Required for JSON deserialization, as tuples are serialized as lists.
        if isinstance(self.indices, list):
            object.__setattr__(self, "indices", tuple(self.indices))


@dataclass
class Gains:
    diagonal: float | ParameterUID
    off_diagonal: float | ParameterUID


@dataclass
class RoutedOutput:
    """Output route of Output Router and Adder (RTR)."""

    from_channel: int
    amplitude: float | ParameterUID
    phase: float | ParameterUID


@dataclass
class IO:
    channel: int
    enable: bool | None = None
    modulation: bool | None = None
    offset: float | None | ParameterUID = None
    gains: Gains | None = None
    range: float | None = None
    range_unit: str | None = None
    precompensation: dict[str, dict] | None = None
    lo_frequency: Any = None
    port_mode: str | None = None
    port_delay: Any = None
    scheduler_port_delay: float = 0.0
    delay_signal: float | None = None
    marker_mode: str | None = None
    amplitude: Any = None
    routed_outputs: list[RoutedOutput] = field(default_factory=list)
    enable_output_mute: bool = False


@dataclass
class AWG:
    awg: int | str
    signal_type: SignalType
    # signal id -> channel (cast to str for compat with json) -> port
    signals: dict[str, dict[str, str]] = field(default_factory=dict)

    # receiver (SG instruments)
    source_feedback_register: int | Literal["local"] | None = None
    codeword_bitshift: int | None = None
    codeword_bitmask: int | None = None
    feedback_register_index_select: int | None = None
    command_table_match_offset: int | None = None

    # transmitter (QA instruments)
    target_feedback_register: int | None = None


@dataclass
class Measurement:
    length: int
    channel: int = 0


@dataclass
class Config:
    repetitions: int = 1
    triggering_mode: TriggeringMode = TriggeringMode.DIO_FOLLOWER
    sampling_rate: float | None = None


@dataclass
class Initialization:
    device_uid: str
    device_type: str | None = None
    config: Config = field(default_factory=Config)
    awgs: list[AWG] = field(default_factory=list)
    outputs: list[IO] = field(default_factory=list)
    inputs: list[IO] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    ppchannels: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OscillatorParam:
    id: str
    device_id: str
    channel: int
    signal_id: str
    frequency: float | None = None
    param: str | None = None


@dataclass
class IntegratorAllocation:
    signal_id: str
    device_id: str
    awg: int
    channels: list[int]
    kernel_count: int
    thresholds: list[float] = field(default_factory=lambda: [0.0])


@dataclass
class AcquireLength:
    signal_id: str
    acquire_length: int


@dataclass
class RealtimeExecutionInit:
    device_id: str | None
    awg_id: int | str
    program_ref: str
    nt_step: NtStepKey
    wave_indices_ref: str | None = None
    kernel_indices_ref: str | None = None


@dataclass
class SoftwareVersions:
    target_labone: str
    laboneq: str


@dataclass
class Recipe:
    initializations: list[Initialization] = field(default_factory=list)
    realtime_execution_init: list[RealtimeExecutionInit] = field(default_factory=list)
    oscillator_params: list[OscillatorParam] = field(default_factory=list)
    integrator_allocations: list[IntegratorAllocation] = field(default_factory=list)
    acquire_lengths: list[AcquireLength] = field(default_factory=list)
    simultaneous_acquires: list[dict[str, str]] = field(default_factory=list)
    total_execution_time: float = 0.0
    max_step_execution_time: float = 0.0
    is_spectroscopy: bool = False
    versions: SoftwareVersions = field(default_factory=lambda: SoftwareVersions("", ""))
