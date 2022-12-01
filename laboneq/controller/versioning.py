# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LabOneVersion(Enum):
    UNKNOWN = "unknown"
    V_22_08 = "22.08"
    LATEST = V_22_08

    def __le__(self, other):
        return float(self.value) <= float(other.value)
