# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


from attrs import define

from laboneq.compiler.ir.section_ir import SectionIR


@define(kw_only=True, slots=True)
class MatchIR(SectionIR):
    handle: str | None
    user_register: int | None
    local: bool | None
    prng_sample: str | None
