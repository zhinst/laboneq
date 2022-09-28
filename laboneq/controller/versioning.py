# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LabOneVersion(Enum):
    UNKNOWN = "unknown"
    V_20_06 = "20.06"
    V_20_07 = "20.07"
    V_20_08 = "20.08"
    V_20_09 = "20.09"
    V_20_10 = "20.10"
    V_21_02 = "21.02"
    V_21_08 = "21.08"
    V_22_02 = "22.02"
    V_22_08 = "22.08"
    LATEST = V_22_08

    def __le__(self, other):
        return float(self.value) <= float(other.value)
