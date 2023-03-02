# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Iterator, List, Set, Tuple

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.new_scheduler.utils import ceil_to_grid


@dataclass(frozen=True)
class SectionSchedule(IntervalSchedule):
    right_aligned: bool

    #: The id of the section
    section: str

    #: Tuple of section IDs that must be scheduled before this interval.
    play_after: Tuple[str, ...]

    #: Trigger info
    trigger_output: Set[Tuple[str, int]]  # signal, bit

    def adjust_length(self, new_length: int):
        """Return a new ``SectionSchedule``, copy of self, but with its length adjusted to the new value.

        The alignment is respected. No check is done to verify if the new length is
        long enough to fit the contents."""
        new_length = ceil_to_grid(new_length, self.grid)
        delta = new_length - self.length
        kwargs = {"length": new_length}
        if self.right_aligned:
            children_start = tuple(start + delta for start in self.children_start)
            kwargs["children_start"] = children_start

        return replace(self, **kwargs)

    def adjust_grid(self, new_grid: int):
        """Return a new ``SectionSchedule``, copy of self, but with its grid adjusted to
        the new value. If the new grid is not commensurate with the length, the length
        is also increased. Right-aligned sections have their children moved to the right
        when adjusting the length."""
        new_length = ceil_to_grid(self.length, new_grid)
        delta = new_length - self.length
        kwargs = {"length": new_length, "grid": new_grid}
        if self.right_aligned:
            children_start = tuple(start + delta for start in self.children_start)
            kwargs["children_start"] = children_start
        return replace(self, **kwargs)

    def add_signals(self, signals: Set[str]):
        """Return a new ``SectionSchedule`` with additional signals."""
        new_signals = frozenset(signals.union(self.signals))
        return replace(self, signals=new_signals)

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: CompilerSettings = None,
    ) -> List[Dict]:

        # We'll wrap the child events in the section start and end events
        max_events -= 2

        trigger_set_events = []
        trigger_clear_events = []
        for trigger_signal, bit in self.trigger_output:
            if max_events < 2:
                break
            max_events -= 2
            event = {
                "event_type": EventType.DIGITAL_SIGNAL_STATE_CHANGE,
                "section_name": self.section,
                "bit": bit,
                "signal": trigger_signal,
            }

            trigger_set_events.append({**event, "change": "SET", "time": start})
            trigger_clear_events.append(
                {
                    **event,
                    "change": "CLEAR",
                    "time": start + self.length,
                }
            )

        children_events = self.children_events(
            start, max_events, settings, id_tracker, expand_loops
        )

        start_id = next(id_tracker)
        d = {"section_name": self.section, "chain_element_id": start_id}
        if self.trigger_output is not None and len(self.trigger_output):
            d["trigger_output"] = [
                {"signal_id": signal} for signal, _ in self.trigger_output
            ]

        return [
            {
                "event_type": EventType.SECTION_START,
                "time": start,
                "id": start_id,
                **d,
            },
            *trigger_set_events,
            *[e for l in children_events for e in l],
            *trigger_clear_events,
            {
                "event_type": EventType.SECTION_END,
                "time": start + self.length,
                "id": next(id_tracker),
                **d,
            },
        ]

    def children_events(
        self,
        start: int,
        max_events: int,
        settings: CompilerSettings,
        id_tracker: Iterator[int],
        expand_loops,
        subsection_events=True,
    ) -> List[List[Dict]]:

        if subsection_events:
            # take into account that we'll wrap with subsection events
            max_events -= 2 * len(self.children)

        children_events = super().children_events(
            start, max_events, settings, id_tracker, expand_loops
        )

        # if children_events was cut because max_events was exceeded, pad with empty
        # lists. This is necessary because the PSV requires the subsection events to be
        # present.
        # todo: investigate if this is a bug in the PSV.
        for i in range(len(self.children) - len(children_events)):
            children_events.append([])

        if subsection_events:
            # Wrap child sections in SUBSECTION_START & SUBSECTION_END.
            start_id = next(id_tracker)
            d = {"section_name": self.section, "chain_element_id": start_id}
            for i, child in enumerate(self.children):
                if isinstance(child, SectionSchedule):
                    children_events[i] = [
                        {
                            "event_type": EventType.SUBSECTION_START,
                            "time": self.children_start[i] + start,
                            "subsection_name": child.section,
                            "id": start_id,
                            **d,
                        },
                        *children_events[i],
                        {
                            "event_type": EventType.SUBSECTION_END,
                            "time": self.children_start[i] + child.length + start,
                            "subsection_name": child.section,
                            "id": next(id_tracker),
                            **d,
                        },
                    ]

        return children_events

    def __hash__(self):
        return super().__hash__()
