# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, auto


class HighPassCompensationClearing(Enum):
    """High-pass compensation clearing.

    !!! version-changed "Deprecated in version 2.8"
        This has no effect.
    """

    LEVEL = auto()
    RISE = auto()
    FALL = auto()
    BOTH = auto()
