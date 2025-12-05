# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

from attrs import define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define(kw_only=True, slots=True)
class AcquireGroupSchedule(IntervalSchedule):
    pulses: list[str]
    amplitudes: list[float]
    handle: str
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
        schedule_data.acquire_pulses.setdefault(self.handle, []).append(self)

        return start
