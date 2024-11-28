# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from attrs import asdict, define

from laboneq.compiler.scheduler.section_schedule import SectionSchedule


@define(kw_only=True, slots=True)
class CaseSchedule(SectionSchedule):
    state: int

    @classmethod
    def from_section_schedule(cls, schedule: SectionSchedule, state: int):
        """Down-cast from SectionSchedule."""
        return cls(**asdict(schedule, recurse=False), state=state)


class EmptyBranch(CaseSchedule):
    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        self.length = self.grid
        return start
