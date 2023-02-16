# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, List, Tuple

from laboneq.compiler.common.compiler_settings import CompilerSettings


@dataclass(frozen=True)
class IntervalSchedule:
    """Base class of a scheduled interval.

    Internally, the schedule is represented as a tree. Nodes of the tree are instances
    of :`~.IntervalSchedule`. An `IntervalSchedule` is an abstraction of a 'box' that
    occupies a certain number of signals (`IntervalSchedule.signals`) for a specific
    duration (`IntervalSchedule.length`). In particular, an `IntervalSchedule` may
    represent a single pulse, or a section (with possible subsections). Loops also are
    similarly represented as nested schedules.

    Notably, each `IntervalSchedule` does not store its start time. Instead, the parent
    stores the start times of its children in `self.children_start`. These time stamps
    are relative to the start time of the parent itself, which allows us to move an
    entire sub-schedule (including its children) in time without touching any of its
    attributes. This 'shifting' in time is valid as long as the start time of a sub-
    schedule, in absolute terms, remains on the grid required by the sub-schedule
    (`IntervalSchedule.grid`). For example, if a section must be aligned to the _system
    grid_ of 4 ns, then we are free to move"""

    #: The time grid along which the interval may be scheduled/shifted. Expressed in
    #: tiny samples.
    grid: int

    #: The length of the interval. Expressed in tiny samples.
    length: int

    #: The signals reserved by this interval.
    signals: FrozenSet[str]

    #: The children of this interval.
    children: Tuple[IntervalSchedule, ...]

    #: The start points of the children *relative to the start of the interval itself*.
    children_start: Tuple[int, ...]

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: CompilerSettings = None,
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

    def __hash__(self):
        # Hashing an interval schedule is expensive! We need to recursively hash the
        # children, potentially traversing the entire section tree.
        # Using `IntervalSchedule` as keys in a dictionary, or storing it in a set is
        # most likely a bad idea.
        raise NotImplementedError
