# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import intervaltree

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.scheduler.section_schedule import SectionSchedule


def calculate_preorder_map(
    schedule: IntervalSchedule, preorder_map: Dict, current_depth=0
) -> int:
    max_depth = current_depth
    intervals = intervaltree.IntervalTree()
    if not isinstance(schedule, SectionSchedule):
        return current_depth
    if isinstance(schedule, LoopSchedule):
        # In the PSV, we do not consider the loop and the loop iteration separately
        schedule = schedule.children[0]

    if isinstance(schedule, SectionSchedule):
        # Draw the section on this row
        assert schedule.section not in preorder_map
        preorder_map[schedule.section] = current_depth
        current_depth += 1

        # Recurse on the children
        for i, c in enumerate(schedule.children):
            if not isinstance(c, SectionSchedule):
                continue
            c_start = schedule.children_start[i]
            c_end = c_start + c.length

            if not intervals.overlap(c_start, c_end):
                # Place child in this row
                max_depth = max(
                    max_depth, calculate_preorder_map(c, preorder_map, current_depth)
                )
            else:
                # Place child in next free row
                max_depth = max(
                    max_depth,
                    calculate_preorder_map(c, preorder_map, max_depth + 1),
                )
            if c_start != c_end:
                intervals.addi(c_start, c_end, data=c.section)

    return max_depth
