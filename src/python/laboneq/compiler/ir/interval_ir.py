# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterator, List, Optional, Set

from attrs import define, field

# A deferred value is not really optional, but will initialized later; using this alias,
# we can still use None as sentinel, but express that this property shall not be seen
# as optional after (possibly external) initialization.
Deferred = Optional


@define(kw_only=True, slots=True)
class IntervalIR:
    #: The children of this interval.
    children: List[IntervalIR] = field(factory=list)

    #: The length of the interval. Typically expressed in tiny samples.
    length: Deferred[int] = None

    #: The signals reserved by this interval.
    signals: Set[str] = field(factory=set)

    #: The start points of the children *relative to the start of the interval itself*.
    children_start: List[int] = field(factory=list)

    def iter_children(self) -> Iterator[tuple[int, IntervalIR]]:
        return zip(self.children_start, self.children)
