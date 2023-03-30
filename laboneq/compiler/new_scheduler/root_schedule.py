# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List, Optional

from laboneq.compiler import CompilerSettings
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.new_scheduler.utils import ceil_to_grid


class RootSchedule(IntervalSchedule):
    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: Optional[CompilerSettings] = None,
    ) -> List[Dict]:
        assert self.length is not None
        children_events = self.children_events(
            start, max_events - 2, settings, id_tracker, expand_loops
        )

        return [e for l in children_events for e in l]

    def _calculate_timing(
        self,
        schedule_data,  # type: ignore # noqa: F821
        start: int,
        start_may_change: bool,
    ):
        length = 0
        for child, child_start in zip(
            self.children,
            self.children_start,  # pyright: ignore[reportGeneralTypeIssues]
        ):
            child.calculate_timing(schedule_data, start + child_start, False)
            assert child.length is not None
            length = max(length, child.length)
        self.length = ceil_to_grid(length, self.grid)

    def __hash__(self):
        super().__hash__()
