# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from attrs import define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


@define(kw_only=True, slots=True)
class InitialOffsetVoltageSchedule(IntervalSchedule):
    value: float

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        return start
