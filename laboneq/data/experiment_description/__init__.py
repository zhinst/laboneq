# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

from numpy.typing import ArrayLike

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data import EnumReprMixin
from laboneq.data.calibration import SignalCalibration
from laboneq.data.parameter import Parameter

#
# Enums
#


class AveragingMode(EnumReprMixin, Enum):
    CYCLIC = auto()
    SEQUENTIAL = auto()
    SINGLE_SHOT = auto()


class ExecutionType(EnumReprMixin, Enum):
    NEAR_TIME = auto()
    REAL_TIME = auto()


class RepetitionMode(EnumReprMixin, Enum):
    AUTO = auto()
    CONSTANT = auto()
    FASTEST = auto()


class SectionAlignment(EnumReprMixin, Enum):
    LEFT = auto()
    RIGHT = auto()


#
# Data Classes
#


@dataclass
class Operation:
    """Operation."""

    ...


@dataclass
class SignalOperation(Operation):
    """Operation on a specific signal."""

    #: Unique identifier of the signal for which the operation is executed.
    signal: str = field(default=None)


@dataclass
class ExperimentSignal:
    uid: str = None
    calibration: Optional[SignalCalibration] = None


@dataclass
class Pulse:
    uid: str = None
    can_compress: bool = False


@dataclass
class Section:
    uid: str = None
    alignment: SectionAlignment = None
    execution_type: ExecutionType | None = None
    length: float | None = None
    play_after: list[str | Section] = field(default_factory=list)
    children: list[Operation | Section] = field(default_factory=list)
    trigger: dict = field(default_factory=dict)
    on_system_grid: bool | None = None


@dataclass
class Acquire(SignalOperation):
    handle: str = None
    kernel: Pulse = None
    length: float = None
    pulse_parameters: Optional[Any] = None


@dataclass
class AcquireLoopNt(Section):
    uid: str = None
    averaging_mode: AveragingMode = None
    count: int = None
    execution_type: ExecutionType = None


@dataclass
class AcquireLoopRt(Section):
    uid: str = None
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION
    averaging_mode: AveragingMode = None
    count: int = None
    execution_type: ExecutionType = None
    repetition_mode: RepetitionMode = None
    repetition_time: float = None
    reset_oscillator_phase: bool = None


@dataclass
class Call(Operation):
    func_name: Any = None
    args: Dict = field(default_factory=dict)


@dataclass
class Case(Section):
    uid: str = None
    state: int = None


@dataclass
class Delay(SignalOperation):
    time: Parameter = None
    precompensation_clear: Optional[bool] = None


@dataclass
class Experiment:
    uid: str = None
    signals: List[ExperimentSignal] = field(default_factory=list)
    epsilon: float = None
    sections: List[Section] = field(default_factory=list)
    pulses: List[Pulse] = field(default_factory=list)


@dataclass
class Match(Section):
    uid: str = None
    handle: Optional[str] = None
    user_register: Optional[int] = None
    local: Optional[bool] = None


@dataclass
class PlayPulse(SignalOperation):
    pulse: Pulse = None
    amplitude: float | complex | Parameter = None
    increment_oscillator_phase: float | Parameter = None
    phase: float | Parameter = None
    set_oscillator_phase: float | Parameter = None
    length: float | Parameter = None
    pulse_parameters: dict | None = None
    precompensation_clear: bool | None = None
    marker: dict | Optional = None


@dataclass
class PulseFunctional(Pulse):
    function: str = None
    amplitude: float | Parameter = None
    length: float = None
    pulse_parameters: dict | None = None


@dataclass
class PulseSampledComplex(Pulse):
    samples: ArrayLike = None


@dataclass
class PulseSampledReal(Pulse):
    samples: ArrayLike = None


@dataclass
class Reserve(SignalOperation):
    ...


@dataclass
class Set(Operation):
    path: str = None
    value: Any = None


@dataclass
class Sweep(Section):
    parameters: List[Parameter] = field(default_factory=list)
    reset_oscillator_phase: bool = None
    execution_type: ExecutionType = None
