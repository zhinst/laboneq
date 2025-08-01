# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

from laboneq.compiler.seqc.signatures import PlaybackSignature


class AWGEventType(Enum):
    LOOP_STEP_START = auto()
    LOOP_STEP_END = auto()
    PUSH_LOOP = auto()
    ITERATE = auto()
    RESET_PRECOMPENSATION_FILTERS = auto()
    INITIAL_RESET_PHASE = auto()
    RESET_PHASE = auto()
    SET_OSCILLATOR_FREQUENCY = auto()
    ACQUIRE = auto()
    QA_EVENT = auto()
    TRIGGER_OUTPUT = auto()
    MATCH = auto()
    PLAY_WAVE = auto()
    PLAY_HOLD = auto()
    INIT_AMPLITUDE_REGISTER = auto()
    CHANGE_OSCILLATOR_PHASE = auto()
    SETUP_PRNG = auto()
    PRNG_SAMPLE = auto()
    DROP_PRNG_SAMPLE = auto()
    PPC_SWEEP_STEP_START = auto()
    PPC_SWEEP_STEP_END = auto()


@dataclass
class AWGEvent:
    type: AWGEventType
    start: int | None = None
    end: int | None = None
    priority: int | None = None
    params: dict[str, Any] = field(default_factory=dict)

    def frozen(self) -> frozenset:
        return frozenset(
            (self.type, self.start, self.end, frozenset(self.params.items()))
        )

    @property
    def signature(self) -> PlaybackSignature:
        return self.params["playback_signature"]

    @property
    def maybe_signature(self) -> PlaybackSignature | None:
        return self.params.get("playback_signature")


@dataclass
class AWGSampledEventSequence:
    """Mapping of the AWG timestamp in device samples to the events at that sample.

    Use `.sort()` whenever sorted order is required.
    """

    sequence: dict[int, list[AWGEvent]] = field(default_factory=dict)

    def add(self, ts: int, event: AWGEvent):
        if ts in self.sequence:
            self.sequence[ts].append(event)
        else:
            self.sequence[ts] = [event]

    def merge(self, other: AWGSampledEventSequence):
        for ts, other_events in other.sequence.items():
            if ts in self.sequence:
                self.sequence[ts].extend(other_events)
            else:
                self.sequence[ts] = other_events

    def sort(self):
        self.sequence = {ts: self.sequence[ts] for ts in sorted(self.sequence)}
