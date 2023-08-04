# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from functools import total_ordering

SUPPORT_PRE_V23_06 = True


@total_ordering
class LabOneVersion(Enum):
    UNKNOWN = "unknown"
    V_23_02 = "23.02"
    V_23_06 = "23.06"
    LATEST = V_23_06

    def __eq__(self, other):
        return float(self.value) == float(other.value)

    def __lt__(self, other):
        return float(self.value) < float(other.value)

    @classmethod
    def cast_if_supported(cls, version: str) -> "LabOneVersion":
        try:
            labone_version = LabOneVersion(version)
            if (labone_version < cls.V_23_06) and (not SUPPORT_PRE_V23_06):
                raise ValueError
        except ValueError:
            err_msg = f"Version {version} is not supported by LabOne Q."
            raise ValueError(err_msg)
        return labone_version
