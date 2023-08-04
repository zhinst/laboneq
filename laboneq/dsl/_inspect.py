# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # | for Union

from collections import deque
from dataclasses import dataclass
from typing import Optional

from laboneq.core.types import CompiledExperiment
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.experiment.section import AcquireLoopRt


def _traverse_experiment(exp: Experiment):
    staq = deque([exp.sections[0]])

    while len(staq):
        sec = staq.pop()
        yield sec
        try:
            staq.extendleft(sec.children)
        except AttributeError:
            pass


@dataclass
class ExperimentInspector:
    """A class for inspecting Experiment and CompiledExperiment objects."""

    _exp: Experiment
    _compiled_exp: Optional[CompiledExperiment] = None

    def get_rt_acquire_loop(self) -> AcquireLoopRt:
        """Return the real-time acquire loop object."""
        for node in _traverse_experiment(self._exp):
            if isinstance(node, AcquireLoopRt):
                return node

    def estimate_runtime(self) -> float:
        """Return an estimation of the total runtime of the experiment in seconds.

        DISCLAIMER: This estimation does not include any overhead from network, IO,
        or python runtime.
        """
        if self._compiled_exp is None:
            raise NotImplementedError(
                "Execution time can only be inspected for CompiledExperiment objects."
            )
        exec_time = self._compiled_exp.recipe.total_execution_time

        return exec_time

    @classmethod
    def from_experiment(cls, exp: Experiment | CompiledExperiment):
        if isinstance(exp, CompiledExperiment):
            return cls(_exp=exp.experiment, _compiled_exp=exp)
        else:
            return cls(_exp=exp)


def inspect(exp: Experiment | CompiledExperiment) -> ExperimentInspector:
    return ExperimentInspector.from_experiment(exp)
