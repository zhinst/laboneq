# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides tasks for collecting several experiment results."""

from __future__ import annotations

from laboneq import workflow

from laboneq.workflow import handles
from laboneq.workflow.tasks.run_experiment import RunExperimentResults


@workflow.task(save=False)
def append_result(
    results: list[RunExperimentResults], result: RunExperimentResults
) -> None:
    """Appends result to results.

    Arguments:
        results: list of RunExperimentResults instances
        result: instance of RunExperimentResults to be appended to results
    """
    results.append(result)


@workflow.task
def combine_results(results: list[RunExperimentResults]) -> RunExperimentResults:
    """Combines the results in results into a single RunExperimentResults instance.

    The results are assumed to be instances of RunExperimentResults, with three
    layers of nesting of the form given by `result_handle(q_uid, suffix=suffix)`.

    Args:
        results: list of RunExperimentResults instances to be combined into a single
            instance of RunExperimentResults

    Returns: instance of RunExperimentResults with all the data in the individual
        RunExperimentResults instances in results.
    """
    data = {}
    for res in results:
        q_names = [n for n in res if n not in ["errors", "neartime_callbacks"]]
        for q_uid in q_names:
            state = next(iter(res[handles.result_handle(q_uid)]))
            res_handle = handles.result_handle(q_uid, suffix=state)
            data[res_handle] = res[res_handle]
    return RunExperimentResults(data=data)
