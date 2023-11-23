# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List

from sortedcollections import SortedDict


class AWGEventType(Enum):
    LOOP_STEP_START = auto()
    LOOP_STEP_END = auto()
    PUSH_LOOP = auto()
    ITERATE = auto()
    SEQUENCER_START = auto()
    RESET_PRECOMPENSATION_FILTERS = auto()
    RESET_PRECOMPENSATION_FILTERS_END = auto()
    INITIAL_RESET_PHASE = auto()
    RESET_PHASE = auto()
    SET_OSCILLATOR_FREQUENCY = auto()
    ACQUIRE = auto()
    QA_EVENT = auto()
    TRIGGER_OUTPUT = auto()
    SWITCH_OSCILLATOR = auto()
    MATCH = auto()
    PLAY_WAVE = auto()
    PLAY_HOLD = auto()
    INIT_AMPLITUDE_REGISTER = auto()
    CHANGE_OSCILLATOR_PHASE = auto()
    SEED_PRNG = auto()


@dataclass
class AWGEvent:
    type: AWGEventType
    start: int = None
    end: int = None
    priority: int | None = None
    params: Dict[str, Any] = field(default_factory=dict)

    def frozen(self) -> frozenset:
        return frozenset(
            (self.type, self.start, self.end, frozenset(self.params.items()))
        )


@dataclass
class AWGSampledEventSequence:
    """Ordered mapping of the AWG timestamp in device samples to the events at that sample."""

    sequence: Dict[int, List[AWGEvent]] = field(default_factory=SortedDict)

    def add(self, ts: int, event: AWGEvent):
        events_at_ts = self.sequence.setdefault(ts, [])
        events_at_ts.append(event)

    def merge(self, other: AWGSampledEventSequence):
        for ts, other_events in other.sequence.items():
            events_at_ts = self.sequence.setdefault(ts, [])
            events_at_ts.extend(other_events)

    def has_matching_event(self, predicate: Callable[[AWGEvent], bool]) -> bool:
        return next(
            (
                True
                for sampled_event_list in self.sequence.values()
                for x in sampled_event_list
                if predicate(x)
            ),
            False,
        )
