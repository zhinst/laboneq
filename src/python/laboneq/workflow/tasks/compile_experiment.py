# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to transform a DSL experiment into a compiled experiment."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq import workflow

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
    return session.compile(
        experiment=experiment,
        compiler_settings=opts.compiler_settings,
    )
