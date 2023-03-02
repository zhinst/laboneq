# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math

import numpy as np


@functools.lru_cache(64)
def lcm(a, b):
    return int(np.lcm(a, b))


def round_to_grid(value, grid: int):
    return round(value / grid) * grid


def ceil_to_grid(value, grid: int):
    return math.ceil(value / grid) * grid


def floor_to_grid(value, grid: int):
    return math.floor(value / grid) * grid
