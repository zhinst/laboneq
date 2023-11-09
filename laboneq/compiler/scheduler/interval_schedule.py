# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Optional, Set, TYPE_CHECKING

from attrs import define, field
from laboneq.compiler.scheduler.utils import lcm

if TYPE_CHECKING:
    from laboneq.compiler.scheduler.schedule_data import ScheduleData

# A deferred value is not really optional, but will initialized later; using this alias,
# we can still use None as sentinel, but express that this property shall not be seen
# as optional after (possibly external) initialization.
Deferred = Optional


@define(kw_only=True, slots=True)
class IntervalSchedule:
    """Base class of a scheduled interval.

    Internally, the schedule is represented as a tree. Nodes of the tree are instances
    of :`~.IntervalSchedule`. An `IntervalSchedule` is an abstraction of a 'box' that
    occupies a certain number of signals (`IntervalSchedule.signals`) for a specific
    duration (`IntervalSchedule.length`). In particular, an `IntervalSchedule` may
    represent a single pulse, or a section (with possible subsections). Loops also are
    similarly represented as nested schedules.

    Notably, each `IntervalSchedule` does initially not store its start time. Instead,
    the parent stores the start times of its children in `self.children_start`. These
    time stamps are relative to the start time of the parent itself, which allows us to
    move an entire sub-schedule (including its children) in time without touching any of
    its attributes. This 'shifting' in time is valid as long as the start time of a sub-
    schedule, in absolute terms, remains on the grid required by the sub-schedule
    (`IntervalSchedule.grid`). For example, if a section must be aligned to the _system
    grid_ of 4 ns, then we are free to move. One exception are match sections, which
    need a minimal distance to their acquire event; `_calculate_timing` thus has a
    parameter `start_may_change` to express that the given start time may not be final.

    Gradually, as right-alignments and RepetitionMode.AUTO timings are resolved,
    intervals are notified about the fact that their absolute start position (after the
    trigger) is resolved and thus can, for example, create constraints for acquire/match
    pairs. The absolute time (in case of loops for the first iteration) is stored in
    `absolute_time`. While this property is somewhat redundant, it also serves as a flag
    to indicate that the timing for this subtree has already be determined and thus
    allows for an early stop.
    """

    #: The children of this interval.
    children: List[IntervalSchedule] = field(factory=list)

    #: The time grid along which the interval may be scheduled/shifted. Expressed in
    #: tiny samples.
    grid: int

    #: The time grid along which the interval may be scheduled/shifted, commensurate
    #: with the sequencer rate. Expressed in tiny samples.
    sequencer_grid: Optional[int] = None

    #: The time grid to be used for compressed loops which contain this section.
    #: Expressed in tiny samples.
    compressed_loop_grid: Optional[int] = None

    #: The length of the interval. Expressed in tiny samples
    length: Deferred[int] = None

    #: The signals reserved by this interval.
    signals: Set[str] = field(factory=set)

    #: The start points of the children *relative to the start of the interval itself*.
    children_start: Deferred[List[int]] = None

    #: The absolute start time (since trigger) of the interval in tiny samples.
    absolute_start: Deferred[int] = None

    #: Whether the schedule can be cached, for example, if no match statements are used
    #: in the section which may lead to timing differences.
    cacheable: bool = True

    def __attrs_post_init__(self):
        for child in self.children:
            self.grid = lcm(self.grid, child.grid)
            self.sequencer_grid = (
                lcm(self.sequencer_grid, child.sequencer_grid)
                if self.sequencer_grid is not None or child.sequencer_grid is not None
                else None
            )
            self.compressed_loop_grid = (
                lcm(self.compressed_loop_grid, child.compressed_loop_grid)
                if self.compressed_loop_grid is not None
                or child.compressed_loop_grid is not None
                else None
            )
            # An acquisition escalates the grid of the containing section
            if getattr(child, "is_acquire", False):
                self.grid = lcm(self.grid, self.sequencer_grid)
            if not child.cacheable:
                self.cacheable = False

    def calculate_timing(
        self, schedule_data: ScheduleData, start: int, start_may_change: bool
    ) -> int:
        if self.children_start is not None:
            # We have already calculated the timing.
            return start
        self.children_start = [0] * len(self.children)
        # Timing calculation may find that the suggested start is too early to fulfill
        # constraints; give it a chance to return a better suiting start time
        start = self._calculate_timing(schedule_data, start, start_may_change)
        if not start_may_change:
            self.on_absolute_start_time_fixed(start, schedule_data)
        return start

    def _calculate_timing(self, *_, **__) -> int:
        raise NotImplementedError()

    def on_absolute_start_time_fixed(self, start: int, schedule_data: ScheduleData):
        """Notify schedule that its absolute start time has been determined, for
        example for a child of a right-aligned section"""
        if self.absolute_start is not None:
            assert start == self.absolute_start
            return
        self.absolute_start = start
        assert self.children_start is not None
        for c, s in zip(self.children, self.children_start):
            c.on_absolute_start_time_fixed(start + s, schedule_data)

    def __hash__(self):
        # Hashing an interval schedule is expensive! We need to recursively hash the
        # children, potentially traversing the entire section tree.
        # Using `IntervalSchedule` as keys in a dictionary, or storing it in a set is
        # most likely a bad idea.
        raise NotImplementedError
