# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import enum


class PlayWaveType(enum.Enum):
    PLAY = enum.auto()
    DELAY = enum.auto()
    CASE_EVALUATION = enum.auto()
    EMPTY_CASE = enum.auto()
    INTEGRATION = enum.auto()
