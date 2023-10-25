# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Iterator, List, Optional, Set

from attrs import define, field

from laboneq.compiler.common.compiler_settings import CompilerSettings

# A deferred value is not really optional, but will initialized later; using this alias,
# we can still use None as sentinel, but express that this property shall not be seen
# as optional after (possibly external) initialization.
Deferred = Optional


@define(kw_only=True, slots=True)
class IntervalIR:
    #: The children of this interval.
    children: List[IntervalIR] = field(factory=list)

    #: The length of the interval. Expressed in tiny samples
    length: Deferred[int] = None

    #: The signals reserved by this interval.
    signals: Set[str] = field(factory=set)

    #: The start points of the children *relative to the start of the interval itself*.
    children_start: Deferred[List[int]] = None

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        raise NotImplementedError

    def children_events(
        self,
        start: int,
        max_events: int,
        settings: CompilerSettings,
        id_tracker: Iterator[int],
        expand_loops: bool,
    ) -> List[List[Dict]]:
        event_list_nested = []
        assert self.children_start is not None
        assert self.length is not None
        for child, child_start in zip(self.children, self.children_start):
            if max_events <= 0:
                break

            event_list_nested.append(
                child.generate_event_list(
                    start + child_start,
                    max_events,
                    id_tracker,
                    expand_loops,
                    settings,
                )
            )
            max_events -= len(event_list_nested[-1])
        return event_list_nested
