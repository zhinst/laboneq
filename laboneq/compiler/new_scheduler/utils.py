# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math


def round_to_grid(value, grid: int):
    return round(value / grid) * grid


def ceil_to_grid(value, grid: int):
    return math.ceil(value / grid) * grid


def floor_to_grid(value, grid: int):
    return math.floor(value / grid) * grid
