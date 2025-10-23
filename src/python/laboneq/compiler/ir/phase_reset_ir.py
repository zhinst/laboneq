# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR


@define(kw_only=True, slots=True)
class PhaseResetIR(IntervalIR):
    section: str
    reset_sw_oscillators: bool
