# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class UsageClassification(Enum):
    WITHIN_LIMIT = auto()
    BEYOND_LIMIT = auto()


@dataclass(frozen=True)
class ResourceUsage:
    desc: str
    """A free-form text associated with this usage."""
    usage: UsageClassification | float
    """Usage, as in used / max available. Either the precise number,
    or a simple classification indicating whether it is <= 1.0 or > 1.0"""


def _gt(a: UsageClassification | float, b: UsageClassification | float) -> bool:
    # Favour floats over simple classification, e.g. 1.2 > BEYOND_LIMIT, 0.2 > WITHIN_LIMIT,
    # but BEYOND_LIMIT > 0.2

    if a == b:
        return False
    if isinstance(a, float) and isinstance(b, float) and a <= b:
        return False
    if isinstance(a, float) and a <= 1.0 and b is UsageClassification.BEYOND_LIMIT:
        return False
    if a is UsageClassification.WITHIN_LIMIT:
        return False
    if a is UsageClassification.BEYOND_LIMIT and isinstance(b, float) and b > 1.0:
        return False
    return True


class ResourceUsageCollector:
    def __init__(self):
        self._max: ResourceUsage | None = None

    def add(self, *usage: ResourceUsage):
        for ru in usage:
            if self._max is None or _gt(ru.usage, self._max.usage):
                self._max = ru

    def raise_or_pass(self):
        if self._max is not None and _gt(self._max.usage, 1.0):
            raise ResourceLimitationError(
                f"Exceeded resource limitation: {self._max}.\n",
                self._max.usage if isinstance(self._max.usage, float) else None,
            )


class ResourceLimitationError(Exception):
    def __init__(self, message: str, hint: float | None = None):
        super().__init__(message)
        self.hint = hint
        """Represents the usage of the said resource."""
