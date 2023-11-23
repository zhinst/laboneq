# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Iterator, List

from laboneq.compiler.ir.interval_ir import IntervalIR


class RootIR(IntervalIR):
    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings,
    ) -> List[Dict]:
        assert self.length is not None
        children_events = self.children_events(
            start, max_events - 2, settings, id_tracker, expand_loops
        )

        return [e for l in children_events for e in l]

    def __hash__(self):
        return super().__hash__()
