# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List

from attrs import asdict, define

from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.compiler.scheduler.oscillator_schedule import (
    OscillatorFrequencyStepSchedule,
)
from laboneq.compiler.scheduler.phase_reset_schedule import PhaseResetSchedule
from laboneq.compiler.scheduler.ppc_step_schedule import PPCStepSchedule
from laboneq.compiler.scheduler.schedule_data import ScheduleData
from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.compiler.scheduler.utils import ceil_to_grid
from laboneq.data.compilation_job import ParameterInfo


@define(kw_only=True, slots=True)
class LoopIterationPreambleSchedule(IntervalSchedule):
    def _calculate_timing(
        self,
        schedule_data: ScheduleData,  # type: ignore # noqa: F821
        suggested_start: int,
        start_may_change: bool,
    ) -> int:
        # The loop preamble's internal schedule obeys slightly different rules than a
        # regular section.
        # First, we schedule all PPC sweep steps and oscillator sweep steps. They always
        # can happen in parallel, because they they are not observable (they cannot
        # overlap with any pulses (or even acquisitions).
        # Finally, we insert the phase resets. These do have observable side effects:
        # 1. We must align them to the LO grid of 100 MHz
        # 2. Their timing w.r.t. to the pulses played in the experiment body is observable
        #    as a phase offset.
        # For this reason, we place them on the right of the preamble.

        sweep_end = 0
        children_starts: list[int | None] = [None for _ in self.children]
        for i, child in enumerate(self.children):
            if isinstance(child, (PPCStepSchedule, OscillatorFrequencyStepSchedule)):
                # these we can plop right at the start
                children_starts[i] = 0
                sweep_end = max(sweep_end, child.length)

        sweep_end = ceil_to_grid(sweep_end, self.grid)

        for i, child in enumerate(self.children):
            if children_starts[i] is not None:
                continue
            assert isinstance(child, PhaseResetSchedule)

            # phase resets go after the sweep steps
            children_starts[i] = ceil_to_grid(sweep_end, child.grid)

        self.children_start = children_starts
        self.length = 0
        for child, child_start in zip(self.children, self.children_start):
            self.length = max(self.length, child_start + child.length)
            child.calculate_timing(
                schedule_data, suggested_start + child_start, start_may_change
            )

        assert self.length % self.grid == 0
        return suggested_start


@define(kw_only=True, slots=True)
class LoopIterationSchedule(SectionSchedule):
    """Schedule of a single iteration of a loop (sweep or average)"""

    iteration: int
    sweep_parameters: List[ParameterInfo]
    num_repeats: int
    shadow: bool
    prng_sample: str | None

    def __attrs_post_init__(self):
        # We always "steal" the data from a SectionSchedule which has already done
        # all the hard work in its own __attrs_post_init__().
        pass

    @classmethod
    def from_section_schedule(
        cls,
        schedule: SectionSchedule,
        iteration,
        num_repeats,
        shadow,
        sweep_parameters,
        prng_sample,
    ):
        """Down-cast from SectionSchedule."""
        return cls(
            **asdict(schedule, recurse=False),
            iteration=iteration,
            num_repeats=num_repeats,
            shadow=shadow,
            sweep_parameters=sweep_parameters,
            prng_sample=prng_sample,
        )
