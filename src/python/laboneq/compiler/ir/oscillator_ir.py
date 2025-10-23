# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import List

from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR


@define(kw_only=True, slots=True)
class SetOscillatorFrequencyIR(IntervalIR):
    values: List[tuple[str, float]]  # signal, frequency


@define(kw_only=True, slots=True)
class InitialOscillatorFrequencyIR(IntervalIR):
    values: List[tuple[str, float]]  # signal, frequency


@define(kw_only=True, slots=True)
class InitialLocalOscillatorFrequencyIR(IntervalIR):
    value: float
