# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.utils import ceil_to_grid


class RootSchedule(IntervalSchedule):
    def _calculate_timing(
        self,
        schedule_data,  # type: ignore # noqa: F821
        start: int,
        start_may_change: bool,
    ) -> int:
        length = 0
        assert start_may_change is False
        for child in self.children:
            child.calculate_timing(schedule_data, 0, start_may_change)
            assert child.length is not None
            length = max(length, child.length)
            child.on_absolute_start_time_fixed(0, schedule_data)
        self.length = ceil_to_grid(length, self.grid)
        return start
