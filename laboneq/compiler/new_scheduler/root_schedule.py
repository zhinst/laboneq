# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List

from laboneq.compiler import CompilerSettings
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule


class RootSchedule(IntervalSchedule):
    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: CompilerSettings = None,
    ) -> List[Dict]:
        children_events = self.children_events(
            start, max_events - 2, settings, id_tracker, expand_loops
        )

        return [e for l in children_events for e in l]

    def __hash__(self):
        super().__hash__()
