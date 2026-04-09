# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to run a compiled experiment in a session."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

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


def _combine_results(results: list[Results]) -> Results:
    """Combine a list of `Results` objects into a single `Results` object.

    !!! warning
        Currently, only the `acquired_results` and `execution_errors` fields are
        combined.

    Arguments:
        results: The list of `Results` objects to combine.

    Returns:
        The combined `Results` object.

    TODO:
        - Raise an error when `Results` objects cannot be combined.
        - Combine all fields of the `Results` objects.
    """
    combined = Results()
    for result in results:
        for handle, data in result.acquired_results.items():
            combined.acquired_results[handle] = data
        combined.execution_errors.extend(result.execution_errors)

    return combined


def _validate_inject_results(
    _instance,
    _attribute,
    value: dict[str, list[Path | str] | Literal["no", "emulation", "always"] | bool],
):
    if set(value.keys()) != {"paths", "use", "check_shapes"}:
        raise ValueError(
            "The `inject_results` dictionary must have the keys {'paths', 'use', 'check_shapes'}."
        )

    if not isinstance(value["paths"], list) or not all(
        isinstance(item, (str, Path)) for item in value["paths"]
    ):
        raise TypeError("The `paths` value must be of type `list[Path | str]`.")

    if value["use"] not in {"no", "emulation", "always"}:
        raise ValueError("The `use` value must be one of 'no', 'emulation', 'always'.")

    if not isinstance(value["check_shapes"], bool):
        raise TypeError("The `check_shapes` value must be of type `bool`.")


def _load_results(path: Path) -> Results:
    """Load a `Results` object from a file path.

    Arguments:
        path: The path to load the results from.

    Returns:
        The loaded `Results` object.

    Raises:
        ValueError: If the file path does not point to serialized `Results`.
    """
    try:
        return serializers.load(path)
    except Exception as e:
        raise ValueError(f"Path {path} does not point to serialized Results.") from e


def _combine_results_from_file_paths(paths: list[Path]) -> Results:
    """Load a list of `Results` objects from file paths and combine them.

    Arguments:
        paths: The list of file paths to load from.

    Returns:
        The combined `Results` object.

    Raises:
        ValueError: If no results file paths are provided.
    """
    if not paths:
        raise ValueError("No results file paths provided.")

    return _combine_results([_load_results(p) for p in paths])


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
            "paths": [],
            "use": "emulation",
            "check_shapes": True,
        },
        validators=[_validate_inject_results],
        # We convert `Path` to `str` so that the workflow inputs are JSON-serializable
        converter=lambda v: {**v, "paths": [str(p) for p in v.get("paths", [])]},
        description=(
            "Inject results loaded from a previously serialized LabOne Q run.\n"
            "Keys:\n"
            "  - use (Literal['no', 'emulation', 'always']): Controls when to inject results.\n"
            "      * 'no'        : never inject\n"
            "      * 'emulation' : inject only when running in emulation mode (default)\n"
            "      * 'always'    : inject regardless of mode\n"
            "      Default: 'emulation'\n"
            "  - paths (list[Path | str]): Paths to load results from.\n"
            "      If multiple paths are provided, then the results are combined.\n"
            "      Default: []\n"
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

    if opts.inject_results["paths"] and (
        opts.inject_results["use"] == "always"
        or (
            opts.inject_results["use"] == "emulation"
            and session._connection_state.emulated
        )
    ):
        paths = [Path(p) for p in opts.inject_results["paths"]]
        injected_results = _combine_results_from_file_paths(paths)
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
