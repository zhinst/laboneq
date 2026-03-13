# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to run a compiled experiment in a session."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from laboneq import (
    serializers,
    workflow,
)
from laboneq.core.utilities.attribute_wrapper import AttributeWrapper  # noqa: F401

# Import of AcquiredResult and AttributeWrapper is to support
# laboneq_applications/analysis/plotting_helpers which imports
# it from here.
from laboneq.dsl.result import AcquiredResult, Results  # noqa: F401

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.simple import Session


# Deprecated alias for Results used by laboneq-applications.
# Remove once laboneq-applications has been updated:
RunExperimentResults = Results


def _are_results_compatible(
    results_to_validate: RunExperimentResults,
    target_results: RunExperimentResults,
    check_shapes: bool = True,
) -> bool:
    """Checks that the result to validate has the same structure as the target result.

    Args:
        results_to_validate: The acquired results to validate.
        target_results: The acquired results to compare against.
        check_shapes: Check if the dimensions of the data in both results match.

    Returns:
        Return a boolean indicating whether the result to validate is compatible
        with the target result.
    """
    # are the keys of the result to validate a subset of the keys of the target result?
    keys_match = (
        results_to_validate.acquired_results.keys()
        <= target_results.acquired_results.keys()
    )
    shapes_match = all(
        results_to_validate.acquired_results[key].data.shape
        == target_results.acquired_results[key].data.shape
        for key in results_to_validate.acquired_results
    )
    # return according to setting
    return keys_match and (shapes_match or not check_shapes)


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

    inject_results: dict = workflow.option_field(
        factory=lambda: {
            "path": None,
            "use": "emulation",
            "check_shapes": True,
        },
        description=(
            "Inject results loaded from a previously serialized LabOne Q run.\n"
            "Keys:\n"
            "  - use (str): Controls when to inject results.\n"
            "      * 'no'        : never inject\n"
            "      * 'emulation' : inject only when running in emulation mode (default)\n"
            "      * 'always'    : inject regardless of mode\n"
            "  - path (Path | str | None): Path to the serialized results file to load.\n"
            "      Default: None\n"
            "  - check_shapes (bool): Whether to verify that injected results match the\n"
            "      expected shapes of the experiment results.\n"
            "      Default: True\n"
        ),
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

    if opts.inject_results["path"] is not None and (
        opts.inject_results["use"] == "always"
        or (
            opts.inject_results["use"] == "emulation"
            and session._connection_state.emulated
        )
    ):
        injected_results = serializers.load(opts.inject_results["path"])
        # checking that dimensions are matching using emulated results
        if not _are_results_compatible(
            results, injected_results, opts.inject_results["check_shapes"]
        ):
            warn_string = (
                f"The injected results from '{opts.inject_results['path']}' are not compatible "
                r"with the expected results. "
                r"Please check that the dimensions and structure of the results match."
            )
            raise ValueError(warn_string)

        results = injected_results

    return results
