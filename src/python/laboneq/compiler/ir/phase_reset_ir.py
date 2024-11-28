# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from typing import List, Tuple

from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR


@define(kw_only=True, slots=True)
class PhaseResetIR(IntervalIR):
    section: str
    hw_osc_devices: List[Tuple[str, float]]
    reset_sw_oscillators: bool


@define(kw_only=True, slots=True)
class PhaseIncrementIR(IntervalIR):
    section: str
    value: float
