# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.types.units import Quantity
from laboneq.data import EnumReprMixin
from laboneq.data.parameter import Parameter


class ModulationType(EnumReprMixin, Enum):
    AUTO = "auto"
    HARDWARE = "hardware"
    SOFTWARE = "software"


class PortMode(EnumReprMixin, Enum):
    LF = "lf"
    RF = "rf"


@dataclass
class BounceCompensation:
    delay: float = None
    amplitude: float = None


@dataclass
class Calibration:
    items: dict[str, SignalCalibration] = field(default_factory=dict)


@dataclass
class MixerCalibration:
    uid: str = None
    voltage_offsets: list[float] | None = None
    correction_matrix: list[list[float]] | None = None


@dataclass
class ExponentialCompensation:
    timeconstant: float = None
    amplitude: float = None


@dataclass
class FIRCompensation:
    coefficients: ArrayLike = None

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FIRCompensation):
            return np.allclose(self.coefficients, other.coefficients)
        else:
            return NotImplemented


@dataclass
class HighPassCompensation:
    timeconstant: float = None


@dataclass
class Oscillator:
    uid: str = None
    frequency: float | Parameter = None
    modulation_type: ModulationType = None


@dataclass
class Precompensation:
    uid: str = None
    exponential: list[ExponentialCompensation] | None = None
    high_pass: HighPassCompensation | None = None
    bounce: BounceCompensation | None = None
    FIR: FIRCompensation | None = None


@dataclass
class AmplifierPump:
    uid: str = None
    pump_freq: float | Parameter | None = None
    pump_power: float | Parameter | None = None
    cancellation: bool = True
    alc_engaged: bool = True
    use_probe: bool = False
    probe_frequency: float | Parameter | None = None
    probe_power: float | Parameter | None = None


@dataclass
class OutputRouting:
    source_signal: str
    amplitude: float
    phase: float


@dataclass
class SignalCalibration:
    oscillator: Oscillator | None = None
    local_oscillator_frequency: float | Parameter | None = None
    mixer_calibration: MixerCalibration | None = None
    precompensation: Precompensation | None = None
    port_delay: float | Parameter | None = None
    port_mode: PortMode | None = None
    delay_signal: float | None = None
    voltage_offset: float | None = None
    range: int | float | Quantity | None = None
    threshold: float | list[float] | None = None
    amplitude: float | Parameter | None = None
    amplifier_pump: AmplifierPump | None = None
    output_routing: list[OutputRouting] = field(default_factory=list)
