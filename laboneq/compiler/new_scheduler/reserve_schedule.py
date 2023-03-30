# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List

from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule


class ReserveSchedule(IntervalSchedule):
    @classmethod
    def create(cls, signal, grid):
        return cls(grid=grid, signals={signal})

    def _calculate_timing(self, *_, **__):
        self.length = 0

    def generate_event_list(self, *_, **__) -> List[Dict]:
        assert self.length is not None
        return []

    def __hash__(self):
        super().__hash__()
