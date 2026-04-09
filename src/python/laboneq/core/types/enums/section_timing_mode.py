# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class SectionTimingMode(Enum):
    RELAXED = "relaxed"
    """Rounding to the hardware timing grid is allowed (default)."""

    STRICT = "strict"
    """Rounding to the hardware timing grid is rejected if the rounding distance
    exceeds 1/1000 of the grid step. This catches accidental off-grid values while
    tolerating floating-point representation noise."""
