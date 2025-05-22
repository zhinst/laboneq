# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides tasks for collecting several experiment results."""

from __future__ import annotations

from laboneq import workflow

from laboneq.dsl.result import Results


@workflow.task(save=False)
def append_result(results: list[Results], result: Results) -> None:
    """Appends result to results.

    Arguments:
        results: list of Results instances
        result: instance of Results to be appended to results
    """
    results.append(result)


@workflow.task
def combine_results(results: list[Results]) -> Results:
    """Combines the results in results into a single Results instance.

    The `acquired_results`, `neartime_callback_results` and
    `execution_errors` are combined as follows:

    * `acquired_results` and `neartime_callback_results` are combined
      using a dictionary update so that later results in the list will
      override earlier results if they share identical handles.

      Note that because `acquired_results` and `neartime_callback_results`
      are accessible via the paths of their handles on the combined
      `Result`, combining results with incompatible paths with raise
      an exception.

      For example, it is an error to combine results with the
      handles `a/b/c` and `a/b`, i.e. where one handle is a
      path prefix of another.

    * `execution_errors` are concatenated into a single list.

    Other result attributes are not combined and are not present
    on the combined result.

    Args:
        results:
            List of Results instances to be combined into a single
            instance of Results.

    Returns:
        An instance of Results with all the data in the individual
        Results instances combined.
    """
    acquired_results = {}
    execution_errors = []
    neartime_callback_results = {}
    for res in results:
        acquired_results.update(res.acquired_results)
        execution_errors.extend(res.execution_errors or [])
        neartime_callback_results.update(res.neartime_callback_results or {})
    return Results(
        acquired_results=acquired_results,
        execution_errors=execution_errors,
        neartime_callback_results=neartime_callback_results,
    )
