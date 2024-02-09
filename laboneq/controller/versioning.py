# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from functools import total_ordering


@total_ordering
class LabOneVersion(Enum):
    UNKNOWN = "unknown"
    V_24_01 = "24.01"
    LATEST = V_24_01

    def __eq__(self, other):
        return float(self.value) == float(other.value)

    def __lt__(self, other):
        return float(self.value) < float(other.value)

    @classmethod
    def cast_if_supported(cls, version: str) -> LabOneVersion:
        try:
            labone_version = LabOneVersion(version)
        except ValueError as e:
            err_msg = f"Version {version} is not supported by LabOne Q."
            raise ValueError(err_msg) from e
        return labone_version


class SetupCaps:
    def __init__(self, version: LabOneVersion | None):
        self._version = version or LabOneVersion.LATEST

    @property
    def result_logger_pipelined(self) -> bool:
        return True
