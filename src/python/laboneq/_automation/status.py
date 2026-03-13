# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class AutomationStatus(Enum):
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
    def inactive(cls) -> list["AutomationStatus"]:
        return [cls.EMPTY, cls.DEACTIVATED, cls.DEACTIVATED_FAIL]

    @classmethod
    def active(cls) -> list["AutomationStatus"]:
        return [cls.READY, cls.FAILED, cls.PASSED, cls.MIXED, cls.RUNNING]
