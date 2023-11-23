# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Iterator, List, Set, Tuple

from attrs import define, field

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
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

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None

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

        prng_setup_events = []
        if self.prng_setup is not None and max_events > 0:
            max_events -= 1
            prng_setup_events = [
                {
                    "event_type": EventType.SETUP_PRNG,
                    "time": start,
                    "section_name": self.section,
                    "range": self.prng_setup.range,
                    "seed": self.prng_setup.seed,
                    "id": next(id_tracker),
                }
            ]

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
            *prng_setup_events,
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
        assert self.children_start is not None

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
        for _ in range(len(self.children) - len(children_events)):
            children_events.append([])

        if subsection_events:
            # Wrap child sections in SUBSECTION_START & SUBSECTION_END.
            for i, child in enumerate(self.children):
                if isinstance(child, SectionIR):
                    assert child.length is not None
                    start_id = next(id_tracker)
                    d = {"section_name": self.section, "chain_element_id": start_id}
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
