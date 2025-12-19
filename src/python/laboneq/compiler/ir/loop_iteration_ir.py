# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.compiler.ir.section_ir import SectionIR


@define(kw_only=True, slots=True)
class LoopIterationPreambleIR(IntervalIR):
    pass


@define(kw_only=True, slots=True)
class LoopIterationIR(SectionIR):
    """IR of a single iteration of a loop (sweep or average)"""
