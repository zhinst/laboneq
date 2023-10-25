# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule


class ReserveSchedule(IntervalSchedule):
    @classmethod
    def create(cls, signal, grid):
        return cls(grid=grid, signals={signal})

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        self.length = 0
        return start

    def __hash__(self):
        return super().__hash__()
