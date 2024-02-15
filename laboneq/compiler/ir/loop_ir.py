# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


from attrs import define

from laboneq.compiler.ir.section_ir import SectionIR


@define(kw_only=True, slots=True)
class LoopIR(SectionIR):
    compressed: bool
    iterations: int
