# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from attrs import asdict, define

from laboneq.compiler.ir.section_ir import SectionIR


@define(kw_only=True, slots=True)
class CaseIR(SectionIR):
    state: int

    @classmethod
    def from_section_ir(cls, schedule: SectionIR, state: int):
        """Down-cast from SectionIR."""
        return cls(**asdict(schedule, recurse=False), state=state)
