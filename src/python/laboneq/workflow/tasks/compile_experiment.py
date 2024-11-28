# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to transform a DSL experiment into a compiled experiment."""  # noqa: E501

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.workflow import task

from laboneq.workflow.tasks.common.attribute_wrapper import find_common_prefix

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.dsl.experiment import Experiment, Section
    from laboneq.dsl.session import Session


def _validate_handles(experiment: Experiment) -> None:
    handles: set[str] = set()

    def update_handles(section: Section) -> None:
        handles.update(
            filter(None, (getattr(s, "handle", None) for s in section.children)),
        )

    experiment.accept_section_visitor(update_handles)
    k = find_common_prefix(handles, "/")
    if k is not None:
        raise ValueError(
            f"Handle '{k[0]}' is a prefix of handle '{k[1]}', which is not"
            " allowed, because a results entry cannot contain both data and "
            "another results subtree. Please rename one of the handles.",
        )


@task
def compile_experiment(
    session: Session,
    experiment: Experiment,
    compiler_settings: dict | None = None,
) -> CompiledExperiment:
    """A task to compile the specified experiment for a given setup.

    This task is used to prepare a LabOne Q DSL experiment for execution on a quantum
    processor. It will return the results of a LabOneQ Session.compile() call.

    Args:
        session:
            A calibrated session to compile the experiment for.
        experiment:
            The LabOne Q DSL experiment to compile.
        compiler_settings:
            Optional settings to pass to the compiler.

    Returns:
        [CompiledExperiment][laboneq.core.types.compiled_experiment.CompiledExperiment]
            The `laboneq` compiled experiment.
    """
    try:
        _validate_handles(experiment)
    except ValueError as error:
        raise ValueError(
            "Invalid input. The following issues were detected: " + str(error),
        ) from error
    return session.compile(
        experiment=experiment,
        compiler_settings=compiler_settings,
    )
