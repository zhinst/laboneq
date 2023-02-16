# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field
from numbers import Number

import numpy as np
from numpy.typing import ArrayLike

parameter_id = 0


def parameter_id_generator():
    global parameter_id
    retval = f"par{parameter_id}"
    parameter_id += 1
    return retval


def _compare_nested(a, b):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        if not (isinstance(b, list) or isinstance(b, np.ndarray)):
            return False
        if not len(a) == len(b):
            return False
        return all(map(lambda x: _compare_nested(x[0], x[1]), zip(a, b)))
    return a == b


@dataclass(init=True, repr=True, order=True)
class Parameter(ABC):
    """Parent class for sweep parameters in a LabOne Q Experiment."""

    uid: str = field(default_factory=parameter_id_generator)


@dataclass(init=True, repr=True, order=True)
class SweepParameter(Parameter):
    """An arbitrary sweep parameter."""

    #: An arbitrary numpy array whose values are used as the sweep parameter.
    values: ArrayLike = field(default_factory=lambda x: np.array([]))

    #: The name of the sweep axis for this parameter used in the results. If this
    #: argument is not defined, the uid of the object will be used instead.
    axis_name: str = field(default=None)

    def __eq__(self, other):
        if self is other:
            return True
        return self.axis_name == other.axis_name and _compare_nested(
            self.values, other.values
        )


@dataclass(init=True, repr=True, order=True)
class LinearSweepParameter(Parameter):
    """A linear sweep parameter"""

    #: The starting value of the parameter sweep.
    start: Number = field(default=None)

    #: The final value of the parameter sweep.
    stop: Number = field(default=None)

    #: The number of sweep steps in the parameter sweep.
    count: int = field(default=None)

    #: The name of the sweep axis for this parameter used in the results.
    #:
    #: If this argument is not defined, the uid of the object will be used instead.
    axis_name: str = field(default=None)

    def __post_init__(self):
        if self.count is None or self.start is None or self.stop is None:
            raise RuntimeError(
                f"LinearSweepParameter {self.uid}: one of start, stop, count is None"
            )

    @property
    def values(self):
        return np.linspace(start=self.start, stop=self.stop, num=self.count)
