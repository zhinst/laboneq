# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List

from laboneq.compiler import CompilerSettings
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.utils import ceil_to_grid


class RootSchedule(IntervalSchedule):
    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        assert self.absolute_start is not None
        children_events = self.children_events(
            start, max_events - 2, settings, id_tracker, expand_loops
        )

        return [e for l in children_events for e in l]

    def _calculate_timing(
        self,
        schedule_data,  # type: ignore # noqa: F821
        start: int,
        start_may_change: bool,
    ) -> int:
        length = 0
        for child in self.children:
            child.calculate_timing(schedule_data, 0, False)
            assert child.length is not None
            length = max(length, child.length)
            child.on_absolute_start_time_fixed(0, schedule_data)
        self.length = ceil_to_grid(length, self.grid)
        return start

    def __hash__(self):
        super().__hash__()
