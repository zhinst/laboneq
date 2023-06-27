# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


import dataclasses
from enum import Enum

from laboneq.dsl.dsl_dataclass_decorator import classformatter


def Volt(val):
    return Quantity(val, Unit.volt)


def dBm(val):
    return Quantity(val, Unit.dBm)


# StrEnum is part of Python 3.11. Until then, let's make our own!
# Downside: auto() does not work...
class StrEnum(str, Enum):
    def __str__(self):
        return str.__str__(self)


class Unit(StrEnum):
    volt = "volt"
    dBm = "dBm"


@classformatter
@dataclasses.dataclass(frozen=True)
class Quantity:
    value: float
    unit: Unit
