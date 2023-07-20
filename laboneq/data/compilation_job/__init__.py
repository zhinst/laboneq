# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Union

from numpy.typing import ArrayLike

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data import EnumReprMixin
from laboneq.data.calibration import (
    BounceCompensation,
    ExponentialCompensation,
    FIRCompensation,
    HighPassCompensation,
    PortMode,
)
from laboneq.data.experiment_description import (
    AveragingMode,
    ExecutionType,
    RepetitionMode,
    SectionAlignment,
)


#
# Enums
#
class DeviceInfoType(EnumReprMixin, Enum):
    UHFQA = "uhfqa"
    HDAWG = "hdawg"
    SHFQA = "shfqa"
    SHFSG = "shfsg"
    SHFQC = "shfqc"
    PQSC = "pqsc"


class ReferenceClockSourceInfo(EnumReprMixin, Enum):
    INTERNAL = auto()
    EXTERNAL = auto()


class SignalInfoType(EnumReprMixin, Enum):
    IQ = auto()
    RF = auto()
    INTEGRATION = auto()


#
# Data Classes
#


@dataclass
class ParameterInfo:
    uid: str
    start: float | None = None
    step: float | None = None
    values: ArrayLike | None = None
    axis_name: str | None = None


@dataclass
class DeviceInfo:
    uid: str = None
    device_type: DeviceInfoType = None
    reference_clock: float = None
    reference_clock_source: ReferenceClockSourceInfo = None
    is_qc: bool | None = None


@dataclass
class OscillatorInfo:
    uid: str = None
    frequency: float | ParameterInfo = None
    is_hardware: bool = None


@dataclass
class PulseDef:
    uid: str = None
    function: str | None = None
    length: float = None
    amplitude: float = None
    phase: float = None
    can_compress: bool = False
    increment_oscillator_phase: float = None
    set_oscillator_phase: float = None
    samples: ArrayLike = field(default_factory=list)
    pulse_parameters: dict | None = None


@dataclass
class SectionInfo:
    uid: str = None
    length: float = None
    alignment: SectionAlignment | None = None
    handle: str | None = None
    state: int | None = None
    local: bool | None = None
    count: int = None
    chunk_count: int = 1
    execution_type: ExecutionType | None = None
    averaging_mode: AveragingMode | None = None
    acquisition_type: AcquisitionType | None = None
    repetition_mode: RepetitionMode | None = None
    repetition_time: float | None = None
    reset_oscillator_phase: bool = False
    children: list[SectionInfo] = field(default_factory=list)
    pulses: list[SectionSignalPulse] = field(default_factory=list)
    on_system_grid: bool = None
    trigger: list = field(default_factory=list)
    parameters: list[ParameterInfo] = field(default_factory=list)
    play_after: list[str] = field(default_factory=list)


@dataclass
class MixerCalibrationInfo:
    voltage_offsets: tuple[float, float] = (0.0, 0.0)
    correction_matrix: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.0),
        (0.0, 1.0),
    )


@dataclass
class PrecompensationInfo:
    exponential: list[ExponentialCompensation] | None = None
    high_pass: HighPassCompensation | None = None
    bounce: BounceCompensation | None = None
    FIR: FIRCompensation | None = None


@dataclass
class SignalRange:
    value: float
    unit: str | None


@dataclass
class AmplifierPumpInfo:
    pump_freq: float | ParameterInfo | None = None
    pump_power: float | ParameterInfo | None = None
    cancellation: bool = True
    alc_engaged: bool = True
    use_probe: bool = False
    probe_frequency: float | ParameterInfo | None = None
    probe_power: float | ParameterInfo | None = None


@dataclass
class SignalInfo:
    uid: str = None
    device: DeviceInfo = None
    oscillator: OscillatorInfo | None = None
    channels: list[int] = field(default_factory=list)
    type: SignalInfoType = None
    voltage_offset: float | None = None
    mixer_calibration: MixerCalibrationInfo | None = None
    precompensation: PrecompensationInfo | None = None
    lo_frequency: float | ParameterInfo | None = None
    signal_range: SignalRange | None = None
    port_delay: float | ParameterInfo | None = None
    delay_signal: float | ParameterInfo | None = None
    port_mode: PortMode | None = None
    threshold: float | None = None
    amplitude: float | ParameterInfo | None = None
    amplifier_pump: AmplifierPumpInfo | None = None


@dataclass
class SectionSignalPulse:
    section: SectionInfo = None
    signal: SignalInfo = None
    pulse_def: PulseDef | None = None
    length: float | ParameterInfo | None = None
    amplitude: float | ParameterInfo | None = None
    phase: float | ParameterInfo | None = None
    increment_oscillator_phase: float | ParameterInfo | None = None
    set_oscillator_phase: float | ParameterInfo | None = None
    precompensation_clear: bool | None = None
    pulse_parameters: dict = field(default_factory=dict)
    acquire_params: AcquireInfo = None
    marker: list[Marker] | None = None


@dataclass
class AcquireInfo:
    handle: str
    acquisition_type: str


@dataclass
class Marker:
    marker_selector: str
    enable: bool
    start: float
    length: float
    pulse_id: str


@dataclass
class ExperimentInfo:
    uid: str = None
    signals: list[SignalInfo] = field(default_factory=list)
    sections: list[SectionInfo] = field(default_factory=list)
    section_signal_pulses: list[SectionSignalPulse] = field(
        default_factory=list
    )  # todo: remove
    global_leader_device: DeviceInfo | None = None  # todo: remove
    pulse_defs: list[PulseDef] = field(default_factory=list)


@dataclass
class CompilationJob:
    uid: str = None
    experiment_hash: str = None
    experiment_info: ExperimentInfo = None
