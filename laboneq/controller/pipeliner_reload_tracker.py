# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import copy

from laboneq.controller.util import LabOneQControllerException
from laboneq.data.recipe import RealtimeExecutionInit


def _merge(
    last: RealtimeExecutionInit, update: RealtimeExecutionInit
) -> RealtimeExecutionInit:
    new = copy.deepcopy(last)
    if update.program_ref is not None:
        new.program_ref = update.program_ref
    if update.wave_indices_ref is not None:
        new.wave_indices_ref = update.wave_indices_ref
    return new


class PipelinerReloadTracker:
    def __init__(self):
        self.last_rt_exec_steps: list[RealtimeExecutionInit] = []

    def calc_next_step(
        self,
        pipeliner_job: int,
        rt_exec_step: RealtimeExecutionInit | None,
    ) -> RealtimeExecutionInit:
        """Constructs the current RT job of a pipeliner (PLn) from recipe data + trace from previous NT steps

        Assuming similar sequence of pipeliner jobs for each near-time step, and that any potential
        differences between identical pipeliner jobs across near-time steps would likely be minor
        compared to the changes between the last pipeliner job of the previous near-time step and
        the first pipeliner job of the next near-time step.

        Legend for the table below:
            * *       - Full data must be available from the recipe
            * <       - Inherit from the previous pipeliner job in the same NT step
            * ^       - Inherit from the same pipeliner job of the previous NT step
            * <+, ^+  - Same as above, but also apply any updates from the recipe

        | NT step | PL0 | PL1 | PL2 | Comment |
        |--------:|:---:|:---:|:---:|:--------|
        |       0 |  *  |  <  |  <  | Only 1st pipeliner job data in recipe, subsequent jobs inherit it
        |       1 |  ^  |  ^  |  ^  | No change since previous NT step, inherit previous pipeliner entirely
        |       2 |  ^+ |  <  |  <  | Update from recipe for the 1st pipeliner job, start filling pipeliner again
        |       3 |  ^  |  ^+ |  ^  | Update from recipe for a pipeliner job > 1
        """
        assert pipeliner_job >= 0
        last_rt_exec_steps = self.last_rt_exec_steps
        if rt_exec_step is None:
            # No update from the recipe
            if pipeliner_job < len(last_rt_exec_steps):
                # Reuse respective job from previous NT step pipeliner
                rt_exec_step = last_rt_exec_steps[pipeliner_job]
            elif (
                pipeliner_job == len(last_rt_exec_steps) and len(last_rt_exec_steps) > 0
            ):
                # Reuse previous pipeliner job
                rt_exec_step = last_rt_exec_steps[-1]
                last_rt_exec_steps.append(rt_exec_step)
            else:
                # Unknown previous pipeliner job
                raise LabOneQControllerException(
                    "Internal error: Could not determine the RT execution params."
                )
        else:
            # Update from recipe
            if pipeliner_job == 0:
                # New pipeline and update recipe - construct fresh pipeliner
                if len(last_rt_exec_steps) > 0:
                    rt_exec_step = _merge(last_rt_exec_steps[0], rt_exec_step)
                last_rt_exec_steps.clear()
                last_rt_exec_steps.append(rt_exec_step)
            elif pipeliner_job < len(last_rt_exec_steps):
                # Amend previous NT step pipeline job
                rt_exec_step = _merge(last_rt_exec_steps[pipeliner_job], rt_exec_step)
                last_rt_exec_steps[pipeliner_job] = rt_exec_step
            elif pipeliner_job == len(last_rt_exec_steps):
                # Amend previous pipeline job
                rt_exec_step = _merge(last_rt_exec_steps[-1], rt_exec_step)
                last_rt_exec_steps.append(rt_exec_step)

        return rt_exec_step
