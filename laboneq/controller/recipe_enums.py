# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class SignalType(Enum):
    IQ = "iq"
    SINGLE = "single"
    INTEGRATION = "integration"
    MARKER = "marker"


class ExecutionType(Enum):
    SINGLE = 1
    SWEEP = 2


class RefClkType(Enum):
    _10MHZ = 10
    _100MHZ = 100


class DIOConfigType(Enum):
    ZSYNC_DIO = 1
    HDAWG = 2
    HDAWG_LEADER = 3
    DIO_FOLLOWER_OF_HDAWG_LEADER = 4


class OperationType(Enum):
    ACQUIRE = "acquire"
    USER_FUNC = "user_func"
    SET = "set"


class ReferenceClockSource(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
