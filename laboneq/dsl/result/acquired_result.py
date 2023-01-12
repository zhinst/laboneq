# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

import numpy as np
from numpy.typing import ArrayLike


def _compare_nested(a, b):
    if isinstance(a, list) or isinstance(a, np.ndarray):
        if not (isinstance(b, list) or isinstance(b, np.ndarray)):
            return False
        if not len(a) == len(b):
            return False
        return all(map(lambda x: _compare_nested(x[0], x[1]), zip(a, b)))
    return a == b


@dataclass(init=True, repr=True, order=True)
class AcquiredResult:
    """
    This class represents the results acquired for an 'acquire' event.

    The acquired result is a triple consisting of actual data, axis name(s)
    and one or more axes
    """

    #: A multidimensional numpy array, where each dimension corresponds to a sweep loop
    #: nesting level, the outermost sweep being the first dimension.
    data: ArrayLike = field(default=None)

    #: A list of axis names. Each element may be either a string or a list of strings.
    axis_name: List[Union[str, List[str]]] = field(default=None)

    #: A list of axis grids. Each element may be either a 1D numpy array or a list of
    #: such arrays.
    axis: List[Union[ArrayLike, List[ArrayLike]]] = field(default=None)

    #: A list of axis indices that represent the last measured near-time point. Only
    #: covers outer near-time dimensions.
    last_nt_step: List[int] = field(default=None)

    def __eq__(self, other: AcquiredResult):
        return (
            _compare_nested(self.data, other.data)
            and self.axis_name == other.axis_name
            and _compare_nested(self.axis, other.axis)
        )
