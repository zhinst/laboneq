# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import math


def normalize_phase(phase: float):
    if phase < 0:
        retval = phase + (int(-phase / 2 / math.pi) + 1) * 2 * math.pi
    else:
        retval = phase
    retval = retval % (2 * math.pi)
    return retval
