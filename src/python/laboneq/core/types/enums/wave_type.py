# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class WaveType(Enum):
    COMPLEX = "complex"
    SINGLE = "single"
    DOUBLE = "double"
    IQ = "iq"

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
