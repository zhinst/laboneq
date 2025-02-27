# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List

from attrs import define

from laboneq.compiler.common.swept_hardware_oscillator import SweptOscillator
from laboneq.compiler.ir.interval_ir import IntervalIR


@define(kw_only=True, slots=True)
class SetOscillatorFrequencyIR(IntervalIR):
    section: str
    oscillators: List[SweptOscillator]
    params: List[str]
    values: List[float]
    iteration: int


@define(kw_only=True, slots=True)
class InitialOscillatorFrequencyIR(IntervalIR):
    oscillators: List[SweptOscillator]
    values: List[float]


@define(kw_only=True, slots=True)
class InitialLocalOscillatorFrequencyIR(IntervalIR):
    value: float
