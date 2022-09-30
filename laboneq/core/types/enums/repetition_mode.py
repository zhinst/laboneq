# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class RepetitionMode(Enum):
    FASTEST = "fastest"
    CONSTANT = "constant"
    AUTO = "auto"
