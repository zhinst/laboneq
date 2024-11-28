# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to run a compiled experiment in a session."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from laboneq import workflow
from typing_extensions import TypeAlias

from laboneq.workflow.tasks.common.attribute_wrapper import AttributeWrapper
from laboneq.workflow.tasks.common.classformatter import classformatter

if TYPE_CHECKING:
    from laboneq.core.types import CompiledExperiment
    from laboneq.simple import Results, Session
    from numpy import typing as npt

ErrorList: TypeAlias = list[tuple[list[int], str, str]]


@dataclass
class AcquiredResult:
    """This class represents the results acquired for a single result handle.

    The acquired result consists of actual data, axis name(s) and one or more axes,
    and resembles the structure of a LabOne Q result with the same name.

    Attributes:
        data (ndarray): A multidimensional `numpy` array, where each dimension
            corresponds to a sweep loop nesting level, the outermost sweep being the
            first dimension.
        axis_name (list[str | list[str]]): A list of axis names.
            Each element may be either a string or a list of strings.
        axis (list[ndarray | list[ndarray]]): A list of axis grids.
            Each element may be either a 1D numpy array or a list of such arrays.
    """

    data: npt.NDArray[Any] | None = None
    axis_name: list[str | list[str]] = field(default_factory=list)
    axis: list[npt.NDArray[Any] | list[npt.NDArray[Any]]] = field(default_factory=list)


@classformatter
class RunExperimentResults(AttributeWrapper):
    """The results of running an experiment.

    The results are accessible via dot notation, where the levels are separated by
    slashes in the handle.

    Example:
    ```python
    acquired = AcquiredResult(
        data=numpy.array([1, 2, 3]),
        axis_name=["Amplitude"],
        axis=[numpy.array([0, 1, 2])],
    )
    results = RunExperimentResults(data={"cal_trace/q0/g": acquired})
    assert results.cal_trace.q0.g is acquired
    assert list(results.cal_trace.q0.keys()) == ["g"]
    ```

    Attributes:
        data:
            The extracted sweep results from the experiment. The keys
            are the acquisition handles.
        neartime_callbacks:
            The results of the near-time user callbacks. The keys are the
            names of the near-time callback functions. The values are the
            list of results in execution order.
        errors:
            The errors that occurred during running the experiment. Each
            item in the list is a tuple of
            `(sweep indices, realt-time section uid, error message)`.
    """

    def __init__(
        self,
        data: dict[str, AcquiredResult],
        neartime_callbacks: dict[str, list[Any]] | None = None,
        errors: ErrorList | None = None,
    ):
        super().__init__(data)
        self._neartime_callbacks = neartime_callbacks or {}
        self._errors = errors or []
        self._key_cache.update(["neartime_callbacks", "errors"])

    @property
    def neartime_callbacks(self) -> AttributeWrapper:
        """The results of the near-time user callbacks."""
        return AttributeWrapper(self._neartime_callbacks)

    @property
    def errors(self) -> ErrorList:
        """The errors that occurred during running the experiment."""
        return self._errors

    def __getitem__(self, key: object) -> AttributeWrapper | ErrorList | object:
        if key == "neartime_callbacks":
            return AttributeWrapper(self._neartime_callbacks)
        if key == "errors":
            return self.errors or []
        return super().__getitem__(key)

    def __dir__(self):
        return list(super().__dir__()) + list(self._key_cache)

    def _as_str_dict(self) -> dict[str, Any]:
        return {
            key: (attr._as_str_dict() if isinstance(attr, AttributeWrapper) else attr)
            for key, attr in ((key, getattr(self, key)) for key in self._key_cache)
        } | {
            "neartime_callbacks": self._neartime_callbacks,
            "errors": self.errors,
        }

    def __repr__(self) -> str:
        return (
            f"RunExperimentResults(data={self._data!r}, "
            f"near_time_callbacks={self._neartime_callbacks!r}, errors={self.errors!r},"
            f" path = {self._path!r}, separator={self._separator!r})"
        )


def extract_results(results: Results | None) -> RunExperimentResults:
    """Extract the results from the LabOne Q results.

    Args:
        results: The LabOne Q results to extract the results from.

    Returns:
        The extracted results.

    Example:
        ```python
        from laboneq_library.tasks.run_experiment import extract_results

        laboneq_results = session.run(compiled_experiment)
        extracted_results = extract_results(laboneq_results)
        ```
    """
    if results is None:
        return RunExperimentResults(data={})
    return RunExperimentResults(
        data={
            h: AcquiredResult(data=r.data, axis=r.axis, axis_name=r.axis_name)
            for h, r in results.acquired_results.items()
        },
        neartime_callbacks=results.neartime_callback_results,
        errors=results.execution_errors,
    )


@workflow.task_options
class RunExperimentOptions:
    """Options for the run_experiment task.

    Attributes:
        return_legacy_results:
            Whether to return an instance of the LabOne Q Results instead of an instance
            of RunExperimentResults.
            Default: False.
    """

    return_legacy_results: bool = workflow.option_field(
        False,
        description="Whether to return an instance of the LabOne Q Results instead of "
        "an instance of RunExperimentResults.",
    )


@workflow.task
def run_experiment(
    session: Session,
    compiled_experiment: CompiledExperiment,
    *,
    options: RunExperimentOptions | None = None,
) -> RunExperimentResults | Results:
    """Run the compiled experiment on the quantum processor via the specified session.

    Arguments:
        session: The connected session to use for running the experiment.
        compiled_experiment: The compiled experiment to run.
        options:
            The options for this task as an instance of [RunExperimentOptions].

    Returns:
        The measurement results as ...
            ... the LabOne Q Results class (returned from `Session.run()`) if
                `return_raw_results` is `True`.
            ... an instance of RunExperimentResults if `return_raw_results` is `False`.
    """
    opts = RunExperimentOptions() if options is None else options
    laboneq_results = session.run(compiled_experiment)
    if opts.return_legacy_results:
        return laboneq_results
    return extract_results(laboneq_results)
