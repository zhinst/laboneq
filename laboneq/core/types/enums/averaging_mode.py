# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class AveragingMode(Enum):
    SEQUENTIAL = "sequential"
    CYCLIC = "cyclic"
    SINGLE_SHOT = "single_shot"
