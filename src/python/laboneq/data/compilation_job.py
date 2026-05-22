# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np

from laboneq.data import EnumReprMixin
from laboneq.data.calibration import CancellationSource

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.data.calibration import (
        BounceCompensation,
        ExponentialCompensation,
        FIRCompensation,
        HighPassCompensation,
        PortMode,
    )
    from laboneq.dsl.calibration.oscillator import Oscillator
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.parameter import Parameter


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
    QHUB = "qhub"
    SHFPPC = "shfppc"
    ZQCS = "zqcs"
    NONQC = "nonqc"


class ReferenceClockSourceInfo(EnumReprMixin, Enum):
    INTERNAL = auto()
    EXTERNAL = auto()


class SignalInfoType(EnumReprMixin, Enum):
    IQ = "iq"
    RF = "single"
    INTEGRATION = "integration"


@dataclass
class DeviceInfo:
    uid: str
    device_type: DeviceInfoType
    options: str = field(default_factory=str)
    dev_opts: list[str] = field(default_factory=list)
    reference_clock_source: ReferenceClockSourceInfo | None = None


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
class MixerCalibrationInfo:
    voltage_offsets: tuple[float | Parameter, float | Parameter] = (0.0, 0.0)
    correction_matrix: (
        tuple[
            tuple[float | Parameter, float | Parameter],
            tuple[float | Parameter, float | Parameter],
        ]
        | list[list[float | Parameter]]
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
    from_port: str | None
    amplitude: float | Parameter
    phase: float | Parameter


@dataclass
class PrecompensationInfo:
    exponential: list[ExponentialCompensation] | None = None
    high_pass: HighPassCompensation | None = None
    bounce: BounceCompensation | None = None
    FIR: FIRCompensation | None = None


@dataclass(unsafe_hash=True)
class SignalRange:
    value: float
    unit: str | None


@dataclass
class AmplifierPumpInfo:
    ppc_device: DeviceInfo | None = None
    pump_frequency: float | Parameter | None = None
    pump_power: float | Parameter | None = None
    pump_on: bool = True
    pump_filter_on: bool = True
    cancellation_on: bool = True
    cancellation_phase: float | Parameter | None = None
    cancellation_attenuation: float | Parameter | None = None
    cancellation_source: CancellationSource = CancellationSource.INTERNAL
    cancellation_source_frequency: float | None = None
    alc_on: bool = True
    probe_on: bool = False
    probe_frequency: float | Parameter | None = None
    probe_power: float | Parameter | None = None
    channel: int | None = None


@dataclass
class SignalInfo:
    uid: str
    device_uid: str
    ports: list[str]

    type: SignalInfoType

    oscillator: Oscillator | None = None
    voltage_offset: float | Parameter | None = None
    mixer_calibration: MixerCalibrationInfo | None = None
    precompensation: PrecompensationInfo | None = None
    lo_frequency: float | Parameter | None = None
    signal_range: SignalRange | None = None
    port_delay: float | Parameter | None = None
    delay_signal: float | None = None
    port_mode: PortMode | None = None
    threshold: float | list[float] | None = None
    amplitude: float | Parameter | None = None
    amplifier_pump: AmplifierPumpInfo | None = None
    output_routing: list[OutputRoute] | None = field(default_factory=list)
    automute: bool = False


@dataclass
class ExperimentInfo:
    device_setup_fingerprint: str
    devices: list[DeviceInfo]
    signals: list[SignalInfo]
    # Scheduler Rust integration fields
    src: Experiment | None = field(default=None)
