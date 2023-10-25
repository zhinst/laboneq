# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from attrs import asdict, define

from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.data.compilation_job import ParameterInfo


@define(kw_only=True, slots=True)
class LoopIterationSchedule(SectionSchedule):
    """Schedule of a single iteration of a loop (sweep or average)"""

    iteration: int
    sweep_parameters: List[ParameterInfo]
    num_repeats: int
    shadow: bool

    def __attrs_post_init__(self):
        # We always "steal" the data from a SectionSchedule which has already done
        # all the hard work in its own __attrs_post_init__().
        pass

    @classmethod
    def from_section_schedule(
        cls, schedule: SectionSchedule, iteration, num_repeats, shadow, sweep_parameters
    ):
        """Down-cast from SectionSchedule."""
        return cls(
            **asdict(schedule, recurse=False),
            iteration=iteration,
            num_repeats=num_repeats,
            shadow=shadow,
            sweep_parameters=sweep_parameters,
        )

    def __hash__(self):
        return super().__hash__()
