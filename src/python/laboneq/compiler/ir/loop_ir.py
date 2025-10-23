# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


from attrs import define, field

from laboneq.compiler.ir.section_ir import SectionIR
from laboneq.core.types.enums import AveragingMode
from laboneq.data.compilation_job import ParameterInfo


@define(kw_only=True, slots=True)
class LoopIR(SectionIR):
    compressed: bool
    iterations: int
    averaging_mode: AveragingMode | None
    sweep_parameters: list[ParameterInfo] = field(factory=list)
