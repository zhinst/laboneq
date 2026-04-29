# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class AutomationStatus(Enum):
    ROOT = "root"
    READY = "ready"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    DEACTIVATED = "deactivated"
    EMPTY = "empty"

    def __str__(self):
        return self.value

    @classmethod
    def inactive(cls) -> list["AutomationStatus"]:
        return [cls.DEACTIVATED]

    @classmethod
    def active(cls) -> list["AutomationStatus"]:
        return [cls.READY, cls.RUNNING, cls.PASSED, cls.FAILED]
