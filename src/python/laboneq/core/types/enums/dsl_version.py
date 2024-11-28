# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class DSLVersion(Enum):
    ALPHA = None
    V2_4_0 = "2.4.0"
    V2_5_0 = "2.5.0"
    V3_0_0 = "3.0.0"
    LATEST = "3.0.0"
