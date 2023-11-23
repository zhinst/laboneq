# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from functools import total_ordering


@total_ordering
class LabOneVersion(Enum):
    UNKNOWN = "unknown"
    V_23_06 = "23.06"
    V_23_10 = "23.10"
    LATEST = V_23_10

    def __eq__(self, other):
        return float(self.value) == float(other.value)

    def __lt__(self, other):
        return float(self.value) < float(other.value)

    @classmethod
    def cast_if_supported(cls, version: str) -> "LabOneVersion":
        try:
            labone_version = LabOneVersion(version)
        except ValueError as e:
            err_msg = f"Version {version} is not supported by LabOne Q."
            raise ValueError(err_msg) from e
        return labone_version
