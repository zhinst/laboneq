# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
from collections import defaultdict

from laboneq.controller.recipe_processor import AwgKey
from laboneq.controller.util import LabOneQControllerException
from laboneq.data.recipe import RealtimeExecutionInit


def _merge(last: RealtimeExecutionInit, update: RealtimeExecutionInit):
    new = copy.deepcopy(last)
    if update.seqc_ref is not None:
        new.seqc_ref = update.seqc_ref
    if update.wave_indices_ref is not None:
        new.wave_indices_ref = update.wave_indices_ref
    return new


class PipelinerReloadTracker:
    def __init__(self):
        self.last_rt_exec_steps_per_awg: dict[
            AwgKey, list[RealtimeExecutionInit]
        ] = defaultdict(list)

    def calc_next_step(
        self,
        awg_key: AwgKey,
        pipeline_chunk: int,
        rt_exec_step: RealtimeExecutionInit,
    ) -> tuple[RealtimeExecutionInit, str]:
        """Constructs the current RT chunk of a pipeline (PL) from recipe data + trace from previous NT steps

        Assuming similar sequence of pipeliner jobs for each near-time step, and that any potential
        differences between identical pipeliner jobs across near-time steps would likely be minor
        compared to the changes between the last pipeliner job of the previous near-time step and
        the first pipeliner job of the next near-time step.

        Legend for the table below:
            * *       - Full data must be available from the recipe
            * <       - Inherit from the previous pipeliner chunk in the same NT step
            * ^       - Inherit from the same pipeliner chunk of the previous NT step
            * <+, ^+  - Same as above, but also apply any updates from the recipe

        | NT step | PL0 | PL1 | PL2 | Comment |
        |--------:|:---:|:---:|:---:|:--------|
        |       0 |  *  |  <  |  <  | Only 1st PL step data in recipe, subsequent steps inherit it
        |       1 |  ^  |  ^  |  ^  | No change since previous NT step, inherit previous PL entirely
        |       2 |  ^+ |  <  |  <  | Update from recipe for the 1st PL step, start filling PL again
        |       3 |  ^  |  ^+ |  ^  | Update from recipe for a PL step > 1
        """
        assert pipeline_chunk >= 0
        last_rt_exec_steps = self.last_rt_exec_steps_per_awg[awg_key]
        if rt_exec_step is None:
            # No update from recipe
            if pipeline_chunk < len(last_rt_exec_steps):
                # Reuse respective chunk from previous PL
                rt_exec_step = last_rt_exec_steps[pipeline_chunk]
            elif (
                pipeline_chunk == len(last_rt_exec_steps)
                and len(last_rt_exec_steps) > 0
            ):
                # Reuse previous PL chunk
                rt_exec_step = last_rt_exec_steps[-1]
                last_rt_exec_steps.append(rt_exec_step)
            else:
                # Unknown previous pipeline chunk
                raise LabOneQControllerException(
                    "Internal error: Could not determine the RT execution params."
                )
        else:
            # Update from recipe
            if pipeline_chunk == 0:
                # New pipeline and update recipe - construct fresh PL
                if len(last_rt_exec_steps) > 0:
                    rt_exec_step = _merge(last_rt_exec_steps[0], rt_exec_step)
                last_rt_exec_steps.clear()
                last_rt_exec_steps.append(rt_exec_step)
            elif pipeline_chunk < len(last_rt_exec_steps):
                # Amend previous NT step pipeline chunk
                rt_exec_step = _merge(last_rt_exec_steps[pipeline_chunk], rt_exec_step)
                last_rt_exec_steps[pipeline_chunk] = rt_exec_step
            elif pipeline_chunk == len(last_rt_exec_steps):
                # Amend previous pipeline chunk
                rt_exec_step = _merge(last_rt_exec_steps[-1], rt_exec_step)
                last_rt_exec_steps.append(rt_exec_step)

        return (
            rt_exec_step,
            f"{awg_key.device_uid}_{awg_key.awg_index}_{pipeline_chunk}.seqc",
        )
