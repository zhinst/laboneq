# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto


class HighPassCompensationClearing(Enum):
    LEVEL = auto()
    RISE = auto()
    FALL = auto()
    BOTH = auto()
