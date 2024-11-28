# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq._rust.intervals import IntervalTree
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.loop_iteration_schedule import LoopIterationSchedule
from laboneq.compiler.scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.scheduler.section_schedule import SectionSchedule


def calculate_preorder_map(
    schedule: IntervalSchedule,
    preorder_map: dict[str, int],
    section_children: dict[str, set[str]],
    current_depth=0,
) -> int:
    if not isinstance(schedule, SectionSchedule):
        return current_depth
    max_depth = current_depth
    if isinstance(schedule, LoopSchedule):
        # Normally we only need to look at the first loop iteration to find all the
        # sections. When there are statically resolved branches however, not every
        # iteration may contain all the subsections.
        for child in schedule.children:
            assert isinstance(child, LoopIterationSchedule)
            # In the PSV, we do not consider the loop and the loop iteration separately, so
            # we immediately pass to the children without incrementing the depth.
            max_depth = max(
                max_depth,
                calculate_preorder_map(
                    child, preorder_map, section_children, current_depth
                ),
            )
            if section_children[schedule.section].issubset(preorder_map.keys()):
                break
        else:
            # When we sweep a parameter in near-time (or the pipeliner), a section can
            # legitimately be absent from the schedule we just generated. This is not
            # an error.
            pass
        return max_depth

    intervals = IntervalTree()
    if isinstance(schedule, SectionSchedule):
        # Draw the section on this row
        preorder_map[schedule.section] = current_depth
        current_depth += 1

        # Recurse on the children
        for i, c in enumerate(schedule.children):
            if not isinstance(c, SectionSchedule):
                continue
            c_start = schedule.children_start[i]
            c_end = c_start + c.length

            if not intervals.overlaps_range(c_start, c_end):
                # Place child in this row
                max_depth = max(
                    max_depth,
                    calculate_preorder_map(
                        c, preorder_map, section_children, current_depth
                    ),
                )
            else:
                # Place child in next free row
                max_depth = max(
                    max_depth,
                    calculate_preorder_map(
                        c, preorder_map, section_children, max_depth + 1
                    ),
                )
            if c_start != c_end:
                intervals.addi(c_start, c_end, data=c.section)

    return max_depth
