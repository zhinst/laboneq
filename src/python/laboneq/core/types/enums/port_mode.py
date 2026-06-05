# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class PortMode(Enum):
    LF = "LF"
    RF = "RF"

    @classmethod
    def _missing_(cls, value):
        if isinstance(value, str) and value != value.upper():
            return cls(value.upper())
        return None
