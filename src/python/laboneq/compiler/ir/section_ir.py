# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Set, Tuple

from attrs import define, field

from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.data.compilation_job import PRNGInfo


@define(kw_only=True, slots=True)
class SectionIR(IntervalIR):
    # The id of the section
    section: str

    # Trigger info: signal, bit
    trigger_output: Set[Tuple[str, int]] = field(factory=set)

    # PRNG setup & seed
    prng_setup: PRNGInfo | None = None
