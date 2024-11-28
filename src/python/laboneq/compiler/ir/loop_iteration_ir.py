# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from attrs import define, evolve

from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.compiler.ir.section_ir import SectionIR
from laboneq.data.compilation_job import ParameterInfo


@define(kw_only=True, slots=True)
class LoopIterationPreambleIR(IntervalIR):
    pass


@define(kw_only=True, slots=True)
class LoopIterationIR(SectionIR):
    """IR of a single iteration of a loop (sweep or average)"""

    iteration: int
    sweep_parameters: List[ParameterInfo]
    num_repeats: int
    shadow: bool
    prng_sample: str | None = None

    def compressed_iteration(self, iteration: int):
        """Make a copy of this schedule, but replace ``iteration`` and set the
        ``shadow`` flag."""
        return evolve(self, iteration=iteration, shadow=True)
