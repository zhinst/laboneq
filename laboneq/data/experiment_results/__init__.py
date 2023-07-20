# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from numpy.typing import ArrayLike

from laboneq.core.validators import dicts_equal


@dataclass
class AcquiredResult:
    """
    This class represents the results acquired for an 'acquire' event.

    The acquired result is a triple consisting of actual data, axis name(s)
    and one or more axes
    """

    #: A multidimensional numpy array, where each dimension corresponds to a sweep loop
    #: nesting level, the outermost sweep being the first dimension.
    data: ArrayLike | None = None

    #: A list of axis names. Each element may be either a string or a list of strings.
    axis_name: list[str | list[str]] = field(default_factory=list)

    #: A list of axis grids. Each element may be either a 1D numpy array or a list of
    #: such arrays.
    axis: list[ArrayLike | list[ArrayLike]] = field(default_factory=list)

    #: A list of axis indices that represent the last measured near-time point. Only
    #: covers outer near-time dimensions.
    last_nt_step: list[int] | None = None

    def __eq__(self, other: AcquiredResult):
        return (
            dicts_equal(self.data, other.data)
            and self.axis_name == other.axis_name
            and dicts_equal(self.axis, other.axis)
            and self.last_nt_step == other.last_nt_step
        )


@dataclass
class ExperimentResults:
    uid: str = None

    #: The acquired results, organized by handle.
    acquired_results: dict[str, AcquiredResult] = field(default_factory=dict)

    #: List of the results of each user user function, by name of the function.
    user_func_results: dict[str, list[Any]] = field(default_factory=dict)

    #: Any exceptions that occurred during the execution of the experiment. Entries are
    #: tuples of
    #:
    #: * the indices of the loops where the error occurred,
    #: * the section uid,
    #: * the error message.
    execution_errors: list[tuple[list[int], str, str]] = field(default_factory=list)

    experiment_hash: str = None
    compiled_experiment_hash: str = None
    execution_payload_hash: str = None
