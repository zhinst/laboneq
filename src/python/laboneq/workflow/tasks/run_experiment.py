# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to run a compiled experiment in a session."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from laboneq import workflow

# Import of AcquiredResult and AttributeWrapper is to support
# laboneq_applications/analysis/plotting_helpers which imports
# it from here.
from laboneq.dsl.result import AcquiredResult  # noqa: F401
from laboneq.core.utilities.attribute_wrapper import AttributeWrapper  # noqa: F401
from laboneq.simple import Results

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.simple import Session


# Deprecated alias for Results used by laboneq-applications.
# Remove once laboneq-applications has been updated:
RunExperimentResults = Results


@workflow.task_options
class RunExperimentOptions:
    """Options for the run_experiment task.

    Attributes:
        include_results_metadata:
            Passed to `Session.run` to specify whether the `.device_setup`,
            `.experiment` and `.compiled_experiment` attributes of the
            result should be populated.
            Default: False.
        return_legacy_results:
            Alias for `include_results_metadata`. Deprecated.
            Default: False.

    !!! version-added "Added in version 2.52.0"
        The option `include_results_metadata` was added.

    !!! version-changed "Deprecated in version 2.52.0"
        A new unified Results class was introduced to LabOne Q. Previously
        `return_legacy_results` specified whether to return the LabOne Q
        `Results` class or its own `RunExperimentResults` class. Now
        the unified `Results` class is always returned but the value of
        `return_legacy_results` determines via the `Session` which
        attributes should be set on the results and is equivalent to
        the new `include_results_metadata`.
    """

    include_results_metadata: bool = workflow.option_field(
        False,
        description="Passed to `Session.run` to specify whether"
        " the `.device_setup`, `.experiment` and `.compiled_experiment`"
        " attributes of the result should be populated.",
    )

    return_legacy_results: bool = workflow.option_field(
        False,
        description="Alias for `include_results_metadata`. Deprecated.",
    )


@workflow.task
def run_experiment(
    session: Session,
    compiled_experiment: CompiledExperiment,
    *,
    options: RunExperimentOptions | None = None,
) -> Results:
    """Run the compiled experiment on the quantum processor via the specified session.

    Arguments:
        session: The connected session to use for running the experiment.
        compiled_experiment: The compiled experiment to run.
        options:
            The options for this task as an instance of [RunExperimentOptions].

    Returns:
        The measurement results.
    """
    opts = RunExperimentOptions() if options is None else options
    if opts.return_legacy_results:
        warnings.warn(
            "The 'return_legacy_results' option is deprecated."
            " Use the 'include_results_metadata' option instead.",
            FutureWarning,
            stacklevel=2,
        )

    include_results_metadata = (
        opts.include_results_metadata or opts.return_legacy_results
    )
    results = session.run(
        compiled_experiment,
        include_results_metadata=include_results_metadata,
    )
    return results
