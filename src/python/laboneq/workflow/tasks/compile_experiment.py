# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to transform a DSL experiment into a compiled experiment."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow
from laboneq.core.types.enums import AveragingMode
from laboneq.dsl.device.instruments import ZQCS
from laboneq.dsl.experiment.section import AcquireLoopRt

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.session import Session


@workflow.task_options
class CompileExperimentOptions:
    """Options for the compile_experiment task.

    Attributes:
        compiler_settings:
            Optional settings to pass to the compiler.
            Default: None.
    """

    compiler_settings: dict | None = workflow.option_field(
        None,
        description="Optional settings to pass to the compiler.",
    )


def _find_acquire_loop(experiment: Experiment) -> AcquireLoopRt:
    """Find acquire loop in the experiment.

    Arguments:
        experiment: The experiment to search in.

    Returns:
        The acquire loop.
    """

    def _search(sections):
        for section in sections:
            if isinstance(section, AcquireLoopRt):
                return section
            found = _search(section.sections)
            if found is not None:
                return found
        return None

    result = _search(experiment.sections)
    if result is None:
        raise ValueError("No acquire loop found in the experiment.")
    return result


@workflow.task
def compile_experiment(
    session: Session,
    experiment: Experiment,
    options: CompileExperimentOptions | None = None,
) -> CompiledExperiment:
    """A task to compile the specified experiment for a given setup.

    This task is used to prepare a LabOne Q DSL experiment for execution on a quantum
    processor. It will return the results of a LabOneQ Session.compile() call.

    Args:
        session:
            A calibrated session to compile the experiment for.
        experiment:
            The LabOne Q DSL experiment to compile.
        options:
            The options for this task as an instance of [CompileExperimentOptions].

    Returns:
        [CompiledExperiment][laboneq.core.types.compiled_experiment.CompiledExperiment]
            The `laboneq` compiled experiment.
    """
    opts = CompileExperimentOptions() if options is None else options

    # TODO: remove this once we have a better solution for the AveragingMode in ZQCS
    if any(isinstance(x, ZQCS) for x in session.device_setup.instruments):
        acquire = _find_acquire_loop(experiment)
        zqcs_requested_averaging_mode = acquire.averaging_mode
        acquire.averaging_mode = AveragingMode.SINGLE_SHOT
    else:
        zqcs_requested_averaging_mode = None

    compiled_experiment = session.compile(
        experiment=experiment,
        compiler_settings=opts.compiler_settings,
    )
    compiled_experiment._zqcs_requested_averaging_mode = zqcs_requested_averaging_mode

    return compiled_experiment
