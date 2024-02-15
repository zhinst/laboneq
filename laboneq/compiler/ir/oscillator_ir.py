# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List

from attrs import define

from laboneq.compiler.common.swept_hardware_oscillator import SweptHardwareOscillator
from laboneq.compiler.ir.interval_ir import IntervalIR


@define(kw_only=True, slots=True)
class OscillatorFrequencyStepIR(IntervalIR):
    section: str
    oscillators: List[SweptHardwareOscillator]
    params: List[str]
    values: List[float]
    iteration: int
