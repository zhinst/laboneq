# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from abc import ABC

from numbers import Number
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
    uid: str = field(default_factory=parameter_id_generator)


@dataclass(init=True, repr=True, order=True)
class SweepParameter(Parameter):
    values: ArrayLike = field(default_factory=lambda x: np.array([]))
    axis_name: str = field(default=None)

    def __eq__(self, other):
        if self is other:
            return True
        return self.axis_name == other.axis_name and _compare_nested(
            self.values, other.values
        )


@dataclass(init=True, repr=True, order=True)
class LinearSweepParameter(Parameter):
    start: Number = field(default=None)
    stop: Number = field(default=None)
    count: int = field(default=None)
    axis_name: str = field(default=None)

    def __post_init__(self):
        if self.count is None or self.start is None or self.stop is None:
            raise RuntimeError(
                f"LinearSweepParameter {self.uid}: one of start, stop, count is None"
            )

    @property
    def values(self):
        return np.linspace(start=self.start, stop=self.stop, num=self.count)
