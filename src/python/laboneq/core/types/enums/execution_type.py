# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ExecutionType(Enum):
    NEAR_TIME = "controller"
    REAL_TIME = "hardware"
