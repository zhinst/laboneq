# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum


class AWGSignalType(Enum):
    SINGLE = "single"  # Only one channel is played
    DOUBLE = "double"  # Two independent channels
    IQ = "iq"  # Two channels form an I/Q signal
    MULTI = "multi"  # Multiple logical channels mixed

    def __repr__(self):
        cls_name = self.__class__.__name__
        return f"{cls_name}.{self.name}"
