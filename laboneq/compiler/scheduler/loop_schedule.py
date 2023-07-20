# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

from attrs import asdict, define

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.scheduler.loop_iteration_schedule import LoopIterationSchedule
from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.compiler.scheduler.utils import ceil_to_grid, lcm
from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.core.types.enums.repetition_mode import RepetitionMode
from laboneq.data.compilation_job import ParameterInfo


@define(kw_only=True, slots=True)
class LoopSchedule(SectionSchedule):
    compressed: bool
    sweep_parameters: List[ParameterInfo]
    iterations: int
    repetition_mode: Optional[RepetitionMode]
    repetition_time: Optional[int]

    def _calculate_timing(
        self, schedule_data: ScheduleData, loop_start: int, start_may_change: bool  # type: ignore # noqa: F821
    ) -> int:
        adjusted_rep_time = (
            None
            if self.repetition_time is None
            else ceil_to_grid(self.repetition_time, self.grid)
        )

        def check_repetition_time(child_length):
            if self.repetition_mode == RepetitionMode.CONSTANT:
                assert adjusted_rep_time is not None
                if child_length > adjusted_rep_time:
                    raise LabOneQException(
                        "Specified repetition time "
                        f"({self.repetition_time*schedule_data.TINYSAMPLE*1e6:.3f} us) "
                        f"is insufficient to fit the content of '{self.section}', "
                        f"iteration {i} "
                        f"({child_length*schedule_data.TINYSAMPLE*1e6:.3f} us)"
                    )

        if self.compressed:
            assert len(self.children) == 1
            assert isinstance(self.children[0], LoopIterationSchedule)
            self.children[0].calculate_timing(
                schedule_data, loop_start, start_may_change
            )
            length = self.children[0].length
            assert length is not None
            check_repetition_time(length)
            if adjusted_rep_time is not None:
                length = adjusted_rep_time
            grid = self.grid
            if self.compressed_loop_grid is not None:
                grid = lcm(grid, self.compressed_loop_grid)
            length = ceil_to_grid(length, grid)
            self.children[0].adjust_length(length)
            self.children_start = [0]
            self.length = length * self.iterations  # type: ignore
        else:
            assert (
                self.repetition_time is not None
                or self.repetition_mode != RepetitionMode.CONSTANT
            )
            repetition_mode = self.repetition_mode or RepetitionMode.FASTEST
            self.children_start = []
            longest = 0
            current_start = 0
            for i, c in enumerate(self.children):
                assert isinstance(c, LoopIterationSchedule)
                self.children_start.append(current_start)
                c.calculate_timing(
                    schedule_data,
                    loop_start + current_start,
                    start_may_change or self.repetition_mode == RepetitionMode.AUTO,
                )
                assert c.length is not None
                if repetition_mode == RepetitionMode.FASTEST:
                    length = ceil_to_grid(c.length, self.grid)
                    c.adjust_length(length)
                    current_start += length
                elif repetition_mode == RepetitionMode.CONSTANT:
                    check_repetition_time(c.length)
                    assert adjusted_rep_time is not None
                    c.adjust_length(adjusted_rep_time)
                    assert c.length == adjusted_rep_time
                    current_start += adjusted_rep_time
                else:  # repetition_mode == RepetitionMode.AUTO
                    length = ceil_to_grid(c.length, self.grid)
                    current_start += length  # preliminary
                    if longest < length:
                        longest = length
            if repetition_mode == RepetitionMode.AUTO:
                for c in self.children:
                    assert isinstance(c, SectionSchedule)
                    c.adjust_length(longest)
                self.children_start = [longest * i for i in range(len(self.children))]
            self._calculate_length(schedule_data)
        return loop_start

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.children_start is not None
        assert self.length is not None
        assert self.absolute_start is not None

        # We'll later wrap the child events in some extra events, see below.
        max_events -= 3

        if not self.compressed:  # unrolled loop
            children_events = list(
                self.children_events(
                    start,
                    max_events,
                    settings,
                    id_tracker,
                    expand_loops,
                    subsection_events=False,
                )
            )
            for iteration, event_list in enumerate(children_events):
                for e in event_list:
                    if "loop_iteration" not in e:
                        e["loop_iteration"] = f"{self.section}_{iteration}"
        else:
            children_events = [
                self.children[0].generate_event_list(
                    start + self.children_start[0],
                    max_events,
                    id_tracker,
                    expand_loops,
                    settings,
                )
            ]
            iteration_event = children_events[0][-1]
            assert iteration_event["event_type"] == EventType.LOOP_ITERATION_END
            iteration_event["compressed"] = True
            if expand_loops:
                prototype = self.children[0]
                assert prototype.length is not None
                assert isinstance(prototype, LoopIterationSchedule)
                iteration_start = start
                for iteration in range(1, self.iterations):
                    max_events -= len(children_events[-1])
                    if max_events <= 0:
                        break
                    iteration_start += prototype.length
                    shadow_iteration = prototype.compressed_iteration(iteration)
                    children_events.append(
                        shadow_iteration.generate_event_list(
                            iteration_start,
                            max_events,
                            id_tracker,
                            expand_loops,
                            settings,
                        )
                    )

        for child_list in children_events:
            for e in child_list:
                if "nesting_level" in e:
                    # todo: pass the current level as an argument, rather than
                    #  incrementing after the fact
                    e["nesting_level"] += 1

        start_id = next(id_tracker)
        d = {
            "section_name": self.section,
            "nesting_level": 0,
            "chain_element_id": start_id,
        }
        return [
            {"event_type": EventType.SECTION_START, "time": start, "id": start_id, **d},
            *[e for l in children_events for e in l],
            {"event_type": EventType.LOOP_END, "time": start + self.length, **d},
            {"event_type": EventType.SECTION_END, "time": start + self.length, **d},
        ]

    def __hash__(self):
        return super().__hash__()

    @classmethod
    def from_section_schedule(
        cls,
        schedule: SectionSchedule,
        compressed: bool,
        sweep_parameters: List[ParameterInfo],
        iterations: int,
        repetition_mode: Optional[RepetitionMode],
        repetition_time: Optional[int],
    ):
        """Down-cast from SectionSchedule."""
        return cls(
            **asdict(schedule, recurse=False),
            compressed=compressed,
            sweep_parameters=sweep_parameters,
            iterations=iterations,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
        )  # type: ignore
