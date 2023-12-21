# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Iterator, List

from attrs import define

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.ir.section_ir import SectionIR


@define(kw_only=True, slots=True)
class MatchIR(SectionIR):
    handle: str | None
    user_register: int | None
    local: bool | None
    prng_sample: str | None

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        events = super().generate_event_list(
            start, max_events, id_tracker, expand_loops, settings
        )
        if len(events) == 0:
            return []
        section_start_event = events[0]
        assert section_start_event["event_type"] == EventType.SECTION_START
        if self.handle is not None:
            section_start_event["handle"] = self.handle
            section_start_event["local"] = self.local
        if self.user_register is not None:
            section_start_event["user_register"] = self.user_register
        if self.prng_sample is not None:
            section_start_event["prng_sample"] = self.prng_sample

        return events

    def __hash__(self):
        return super().__hash__()
