# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.data import EnumReprMixin
from laboneq.data.experiment_description import (
    AveragingMode,
    ExecutionType,
    RepetitionMode,
    SectionAlignment,
)

from laboneq.data.calibration import CancellationSource

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from laboneq.core.types.enums.acquisition_type import AcquisitionType
    from laboneq.data.calibration import (
        BounceCompensation,
        ExponentialCompensation,
        FIRCompensation,
        HighPassCompensation,
        PortMode,
    )
    from laboneq.executor.executor import Statement


#
# Enums
#
class DeviceInfoType(EnumReprMixin, Enum):
    UHFQA = "uhfqa"
    HDAWG = "hdawg"
    SHFQA = "shfqa"
    SHFSG = "shfsg"
    PQSC = "pqsc"
    QHUB = "qhub"
    SHFPPC = "shfppc"
    PRETTYPRINTERDEVICE = "prettyprinterdevice"
    NONQC = "nonqc"


class ReferenceClockSourceInfo(EnumReprMixin, Enum):
    INTERNAL = auto()
    EXTERNAL = auto()


class SignalInfoType(EnumReprMixin, Enum):
    IQ = "iq"
    RF = "single"
    INTEGRATION = "integration"


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

    # auto generated __eq__ fails to compare array-likes correctly
    def __eq__(self, other: object) -> bool:
        if isinstance(other, ParameterInfo):
            if (self.values is None) != (other.values is None):
                return False
            try:
                values_equal = (
                    self.values is None and other.values is None
                ) or np.allclose(self.values, other.values)
            except ValueError:
                # numpy raises ValueError if the shapes mismatch
                return False
            return (self.uid, self.start, self.step, self.axis_name) == (
                other.uid,
                other.start,
                other.step,
                other.axis_name,
            ) and values_equal
        return NotImplemented


@dataclass
class FollowerInfo:
    device: DeviceInfo
    port: int


@dataclass
class DeviceInfo:
    uid: str = None
    device_type: DeviceInfoType = None
    reference_clock_source: ReferenceClockSourceInfo | None = None
    is_qc: bool | None = None
    followers: list[FollowerInfo] = field(default_factory=list)


@dataclass
class OscillatorInfo:
    uid: str = None
    frequency: float | ParameterInfo = None
    is_hardware: bool = None


@dataclass()
class PulseDef:
    uid: str = None
    function: str | None = None
    length: float = None
    amplitude: float = 1.0
    phase: float = 0.0
    can_compress: bool = False
    samples: ArrayLike | None = None

    # auto generated __eq__ fails to compare array-likes correctly
    def __eq__(self, other: object) -> bool:
        if isinstance(other, PulseDef):
            if (self.samples is None) != (other.samples is None):
                return False
            samples_equal = (
                self.samples is None and other.samples is None
            ) or np.allclose(self.samples, other.samples)
            return (
                self.uid,
                self.function,
                self.length,
                self.amplitude,
                self.phase,
                self.can_compress,
            ) == (
                other.uid,
                other.function,
                other.length,
                other.amplitude,
                other.phase,
                other.can_compress,
            ) and samples_equal
        return NotImplemented

    def __hash__(self) -> int:
        samples_tuple = tuple(self.samples) if self.samples is not None else None

        return hash(
            (
                self.uid,
                self.function,
                self.length,
                self.amplitude,
                self.phase,
                self.can_compress,
                samples_tuple,
            )
        )


@dataclass
class PRNGInfo:
    range: int
    seed: int


@dataclass
class SectionInfo:
    uid: str = None
    length: float = None
    alignment: SectionAlignment | None = None
    on_system_grid: bool = None

    children: list[SectionInfo] = field(default_factory=list)
    pulses: list[SectionSignalPulse] = field(default_factory=list)

    signals: list[SignalInfo] = field(default_factory=list)

    match_handle: str | None = None
    match_user_register: int | None = None
    match_prng_sample: str | None = None
    match_sweep_parameter: ParameterInfo | None = None
    local: bool | None = None
    state: int | None = None
    prng: PRNGInfo | None = None
    prng_sample: str | None = None

    count: int | None = None  # 'None' means 'not a loop'
    chunk_count: int = 1
    execution_type: ExecutionType | None = None
    averaging_mode: AveragingMode | None = None
    repetition_mode: RepetitionMode | None = None
    repetition_time: float | None = None

    acquisition_type: AcquisitionType | None = None
    reset_oscillator_phase: bool = False
    triggers: list[dict[str, Any]] = field(default_factory=list)
    parameters: list[ParameterInfo] = field(default_factory=list)
    play_after: list[str] = field(default_factory=list)


@dataclass
class MixerCalibrationInfo:
    voltage_offsets: tuple[float | ParameterInfo, float | ParameterInfo] = (0.0, 0.0)
    correction_matrix: (
        tuple[
            tuple[float | ParameterInfo, float | ParameterInfo],
            tuple[float | ParameterInfo, float | ParameterInfo],
        ]
        | list[list[float | ParameterInfo]]
    ) = (
        (1.0, 0.0),
        (0.0, 1.0),
    )


@dataclass
class OutputRoute:
    """Output route of Output Router and Adder (RTR).

    Attributes:
        to_channel: Target channel of the Output Router.
        from_channel: Source channel of the Output Router.
        to_signal: Target channel's Experiment signal UID.
        from_signal: Source channel's Experiment signal UID.
        amplitude: Amplitude scaling of the source signal.
        phase: Phase shift of the source signal.
    """

    to_channel: int
    from_channel: int
    to_signal: str
    from_signal: str | None
    amplitude: float | ParameterInfo
    phase: float | ParameterInfo


@dataclass
class PrecompensationInfo:
    exponential: list[ExponentialCompensation] | None = None
    high_pass: HighPassCompensation | None = None
    bounce: BounceCompensation | None = None
    FIR: FIRCompensation | None = None
    computed_delay_samples: int | None = None


@dataclass
class SignalRange:
    value: float
    unit: str | None


@dataclass
class AmplifierPumpInfo:
    ppc_device: DeviceInfo | None = None
    pump_frequency: float | ParameterInfo | None = None
    pump_power: float | ParameterInfo | None = None
    pump_on: bool = True
    pump_filter_on: bool = True
    cancellation_on: bool = True
    cancellation_phase: float | ParameterInfo | None = None
    cancellation_attenuation: float | ParameterInfo | None = None
    cancellation_source: CancellationSource = CancellationSource.INTERNAL
    cancellation_source_frequency: float | None = None
    alc_on: bool = True
    probe_on: bool = False
    probe_frequency: float | ParameterInfo | None = None
    probe_power: float | ParameterInfo | None = None
    channel: int | None = None


@dataclass
class SignalInfo:
    uid: str
    device: DeviceInfo = None
    oscillator: OscillatorInfo | None = None
    channels: list[int] = field(default_factory=list)
    channel_to_port: dict[str, str] = field(default_factory=dict)
    type: SignalInfoType = None
    voltage_offset: float | None = None
    mixer_calibration: MixerCalibrationInfo | None = None
    precompensation: PrecompensationInfo | None = None
    lo_frequency: float | ParameterInfo | None = None
    signal_range: SignalRange | None = None
    port_delay: float | ParameterInfo | None = None
    delay_signal: float | ParameterInfo | None = None
    port_mode: PortMode | None = None
    threshold: float | list[float] | None = None
    amplitude: float | ParameterInfo | None = None
    amplifier_pump: AmplifierPumpInfo | None = None
    kernel_count: int | None = None
    output_routing: list[OutputRoute] | None = field(default_factory=list)
    automute: bool = False


@dataclass
class SectionSignalPulse:
    signal: SignalInfo = None
    pulse: PulseDef | None = None
    length: float | ParameterInfo | None = None
    offset: float | ParameterInfo | None = None
    amplitude: float | ParameterInfo | None = None
    phase: float | ParameterInfo | None = None
    increment_oscillator_phase: float | ParameterInfo | None = None
    set_oscillator_phase: float | ParameterInfo | None = None
    precompensation_clear: bool | None = None
    play_pulse_parameters: dict = field(default_factory=dict)
    pulse_pulse_parameters: dict = field(default_factory=dict)
    acquire_params: AcquireInfo = None
    markers: list[Marker] = field(default_factory=list)
    pulse_group: str | None = None


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
    pulse_id: str | None


@dataclass
class ExperimentInfo:
    uid: str = None
    devices: list[DeviceInfo] = field(default_factory=list)
    signals: list[SignalInfo] = field(default_factory=list)
    sections: list[SectionInfo] = field(default_factory=list)
    global_leader_device: DeviceInfo | None = None  # todo: remove
    pulse_defs: list[PulseDef] = field(default_factory=list)


@dataclass
class CompilationJob:
    uid: str = None
    experiment_hash: str = None
    experiment_info: ExperimentInfo = None
    execution: Statement = None
    compiler_settings: dict = None
