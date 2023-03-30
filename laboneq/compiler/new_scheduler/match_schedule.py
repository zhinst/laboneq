# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List, Optional

from attrs import define

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.new_scheduler.case_schedule import CaseSchedule
from laboneq.compiler.new_scheduler.schedule_data import ScheduleData
from laboneq.compiler.new_scheduler.section_schedule import SectionSchedule
from laboneq.core.exceptions.laboneq_exception import LabOneQException


@define(kw_only=True, slots=True)
class MatchSchedule(SectionSchedule):
    handle: str
    local: bool

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        self.cacheable = False

    def _calculate_timing(
        self, schedule_data: ScheduleData, start: int, start_may_change
    ):
        if start_may_change:
            raise LabOneQException(
                f"Match Section '{self.section}' may not be a subsection of a right-aligned section."
            )

        for c in self.children:
            assert isinstance(c, CaseSchedule)
            c.calculate_timing(schedule_data, start, start_may_change)
            # Start of children stays at 0

        self._calculate_length(schedule_data)

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: Optional[CompilerSettings] = None,
    ) -> List[Dict]:
        assert self.length is not None
        events = super().generate_event_list(
            start, max_events, id_tracker, expand_loops, settings
        )
        if len(events) == 0:
            return []
        section_start_event = events[0]
        assert section_start_event["event_type"] == EventType.SECTION_START
        section_start_event["handle"] = self.handle
        section_start_event["local"] = self.local

        return events

    def __hash__(self):
        super().__hash__()
