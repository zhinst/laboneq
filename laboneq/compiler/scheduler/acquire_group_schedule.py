# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

from attrs import define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.data.compilation_job import SectionSignalPulse


@define(kw_only=True, slots=True)
class AcquireGroupSchedule(IntervalSchedule):
    pulses: list[SectionSignalPulse]
    amplitudes: list[float]
    phases: list[float]
    offset: int
    oscillator_frequencies: list[Optional[float]]
    section: str
    play_pulse_params: list[Optional[Dict[str, Any]]]
    pulse_pulse_params: list[Optional[Dict[str, Any]]]

    def _calculate_timing(
        self,
        schedule_data: ScheduleData,  # type: ignore # noqa: F821
        start: int,
        start_may_change: bool,
    ) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None

        valid_pulse = next(
            (
                p
                for p in self.pulses
                if p
                and p.pulse is not None
                and p.acquire_params is not None
                and p.acquire_params.handle
            ),
            None,
        )
        if valid_pulse is not None:
            schedule_data.acquire_pulses.setdefault(
                valid_pulse.acquire_params.handle, []
            ).append(self)

        return start

    def __hash__(self):
        return super().__hash__()
