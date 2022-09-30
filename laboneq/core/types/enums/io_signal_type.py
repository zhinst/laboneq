# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class IOSignalType(Enum):
    I = "I"
    Q = "Q"
    IQ = "IQ"
    RF = "RF"
    SINGLE = "SINGLE"
    LO = "LO"
    DIO = "DIO"
    ZSYNC = "ZSYNC"
