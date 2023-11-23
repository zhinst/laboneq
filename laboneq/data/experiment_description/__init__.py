# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from numpy.typing import ArrayLike

from laboneq.core.types.enums import (
    AveragingMode,
    ExecutionType,
    RepetitionMode,
    SectionAlignment,
)
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.calibration import Calibration
from laboneq.data.parameter import Parameter
from laboneq.data.prng import PRNGSample, PRNG


#
# Enums
#

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
    kernel: Pulse | list[Pulse] | None = None
    length: float = None
    pulse_parameters: Optional[Any] | list[Optional[Any]] = None


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
    averaging_mode: AveragingMode = AveragingMode.CYCLIC
    count: int = None
    execution_type: ExecutionType = ExecutionType.REAL_TIME
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
    sections: List[Section] = field(default_factory=list)
    pulses: List[Pulse] = field(default_factory=list)
    #: Calibration for individual `ExperimentSignal`s.
    calibration: Calibration = field(default_factory=Calibration)


@dataclass
class Match(Section):
    uid: str = None
    handle: str | None = None
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
    marker: dict | None = None


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
class SetNode(Operation):
    path: str = None
    value: Any = None


@dataclass
class Sweep(Section):
    parameters: List[Parameter] = field(default_factory=list)
    reset_oscillator_phase: bool = None
    execution_type: ExecutionType = None
    chunk_count: int = 1


@dataclass
class PrngSetup(Section):
    prng: PRNG = None


@dataclass
class PrngLoop(Section):
    prng_sample: PRNGSample = None
