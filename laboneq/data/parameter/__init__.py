# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from numpy.typing import ArrayLike


@dataclass
class Parameter:
    uid: str = None


@dataclass
class LinearSweepParameter(Parameter):
    uid: str = None
    start: float | complex = None
    stop: float | complex = None
    count: int = None
    axis_name: str = None


@dataclass
class SweepParameter(Parameter):
    values: ArrayLike = None
    axis_name: str = None
    driven_by: list[Parameter] = field(default_factory=list)
