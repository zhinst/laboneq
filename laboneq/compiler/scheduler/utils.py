# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
from typing import Optional, TypeVar

import numpy as np


@functools.lru_cache(64)
def lcm(a, b):
    return int(np.lcm(a or 1, b or 1))


def round_to_grid(value, grid: int):
    return round(value / grid) * grid


def ceil_to_grid(value, grid: int):
    return math.ceil(value / grid) * grid


def floor_to_grid(value, grid: int):
    return math.floor(value / grid) * grid


def to_tinysample(t: float | None, tinysample: float) -> int | None:
    return None if t is None else round(t / tinysample)


T = TypeVar("T")


def assert_valid(obj: Optional[T]) -> T:
    assert obj is not None
    return obj
