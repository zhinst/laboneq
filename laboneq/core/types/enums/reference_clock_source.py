# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class ReferenceClockSource(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
