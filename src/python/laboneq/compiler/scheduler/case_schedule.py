# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from attrs import define

from laboneq.compiler.scheduler.section_schedule import SectionSchedule


@define(kw_only=True, slots=True)
class CaseSchedule(SectionSchedule):
    state: int

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        return super()._calculate_timing(_schedule_data, start, *__, **___)
