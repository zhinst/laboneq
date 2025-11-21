# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Literal


def nearest_multiple(
    a: float,
    b: float,
    rounding: str | Literal["ceil", "round", "floor"] = "round",
) -> float:
    """Return the multiple of `b` closest to `a`.

    Performs `op(a/b) * b` where `op` is one of `math.ceil`, `math.round` or
    `math.floor`.

    Args:
        a: Number to round.
        b: Multiplicant.
        rounding: One of ["ceil", "round", "floor"]. Determines `op`.

    """
    if rounding == "ceil":
        return math.ceil(a / b) * b
    elif rounding == "round":
        return round(a / b) * b
    elif rounding == "floor":
        return math.floor(a / b) * b
    else:
        raise ValueError("Unknown rounding.")


def nearest_multiple_floor(a: float, b: float, threshold: float = 1e-12) -> float:
    """For the expression `(a + threshold) = n*b + epsilon`, returns `n*b`
    minimizing epsilon where epsilon is strictly positive and n is a positive
    integer.

    Args:
        a: Number to round.
        b: Multiplicant.
        threshold: Maximum difference allowed between two numbers to consider
            them near. Defaults to `1e-12`.

    """

    assert threshold > 0, f"`threshold` must be positive. ({threshold})"

    return nearest_multiple(a + threshold, b, "floor")


def nearest_multiple_ceil(a: float, b: float, threshold: float = 1e-12) -> float:
    """For the expression `(a - threshold) = n*b - epsilon`, returns `n*b`
    minimizing epsilon where epsilon is strictly positive and n is a positive
    integer.

    Args:
        a: Number to round.
        b: Multiplicant.
        threshold: Maximum difference allowed between two numbers to consider
            them near. Defaults to `1e-12`.

    """

    assert threshold > 0, f"`threshold` must be positive. ({threshold})"

    return nearest_multiple(a - threshold, b, "ceil")
