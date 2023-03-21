# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, Iterator, List

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.new_scheduler.section_schedule import SectionSchedule


@dataclass(frozen=True)
class MatchSchedule(SectionSchedule):
    handle: str
    local: bool

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: CompilerSettings = None,
    ) -> List[Dict]:
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
