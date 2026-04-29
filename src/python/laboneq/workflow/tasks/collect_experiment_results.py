# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides tasks for collecting several experiment results."""

from __future__ import annotations

from laboneq import workflow
from laboneq.dsl.result import Results
from laboneq.dsl.result import combine_results as _combine_results


@workflow.task(save=False)
def append_result(results: list[Results], result: Results) -> None:
    """Appends result to results.

    Arguments:
        results: list of Results instances
        result: instance of Results to be appended to results
    """
    results.append(result)


@workflow.task(save=False)
def combine_results(results: list[Results]) -> Results:
    """Combines a list of results into a single `Results` instance.

    See `laboneq.dsl.result.combine_results` for details on
    how the results are combined.

    Args:
        results:
            List of `Results` instances to be combined into a single
            instance of `Results`.

    Returns:
        An instance of `Results` with all the data in the individual
        `Results` instances combined.
    """
    combined_results = _combine_results(results)
    workflow.save_artifact("combine_results.output", combined_results)
    return combined_results
