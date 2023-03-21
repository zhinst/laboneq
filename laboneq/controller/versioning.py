# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class LabOneVersion(Enum):
    UNKNOWN = "unknown"
    V_23_02 = "23.02"
    LATEST = V_23_02

    def __le__(self, other):
        return float(self.value) <= float(other.value)
