# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class PulseType(Enum):
    GAUSSIAN = "gaussian"
    CONST = "const"
    INTERNAL = "ZI_internal_pulse_1"
