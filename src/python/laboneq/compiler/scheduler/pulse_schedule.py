# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

from attrs import define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define(kw_only=True, slots=True)
class PulseSchedule(IntervalSchedule):
    pulse: str
    amplitude: float
    amp_param_name: str | None = None
    phase: float
    set_oscillator_phase: Optional[float] = None
    increment_oscillator_phase: Optional[float] = None
    incr_phase_param_name: str | None = None
    play_pulse_params: Optional[Dict[str, Any]] = None
    pulse_pulse_params: Optional[Dict[str, Any]] = None
    pulse_params_id: int | None = None
    is_acquire: bool
    markers: Any = None
    # integration length originally specified by the user in case of acquire
    integration_length: int | None
    handle: str | None = None

    def _calculate_timing(
        self,
        schedule_data: ScheduleData,  # type: ignore # noqa: F821
        start: int,
        start_may_change: bool,
    ) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None
        if self.handle:
            schedule_data.acquire_pulses.setdefault(self.handle, []).append(self)
        return start


@define(kw_only=True, slots=True)
class PrecompClearSchedule(IntervalSchedule):
    signal: str

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        self.length = 0
        return start
