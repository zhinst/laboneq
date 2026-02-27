# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


class AutomationElementStatus(Enum):
    ROOT = "root"
    EMPTY = "empty"
    FAILED = "failed"
    READY = "ready"
    PASSED = "passed"
    MIXED = "mixed"
    RUNNING = "running"
    DEACTIVATED = "deactivated"
    DEACTIVATED_FAIL = "deactivated (failure)"

    def __str__(self):
        return self.value

    @classmethod
    def inactive(cls) -> list["AutomationElementStatus"]:
        return [cls.EMPTY, cls.DEACTIVATED, cls.DEACTIVATED_FAIL]

    @classmethod
    def active(cls) -> list["AutomationElementStatus"]:
        return [cls.READY, cls.FAILED, cls.PASSED, cls.MIXED, cls.RUNNING]


@classformatter
@attrs.define(kw_only=True)
class AutomationElement(ABC):
    """An element in the automation framework (i.e. layer or node).

    Attributes:
        key: The automation element key.
        depends_on: A set of automation element dependencies.
    """

    key: str
    depends_on: set[str]
