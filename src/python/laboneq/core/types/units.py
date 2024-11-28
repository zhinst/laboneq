# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from enum import Enum

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter


# StrEnum is part of Python 3.11. Until then, let's make our own!
# Downside: auto() does not work...
class _StrEnum(str, Enum):
    def __str__(self):
        return str.__str__(self)


class Unit(_StrEnum):
    """An Enum of available calibration units."""

    volt = "volt"
    dBm = "dBm"


@classformatter
@dataclasses.dataclass(frozen=True)
class Quantity:
    """A quantity with units.

    Attributes:
        value:
            The value of the quantity.
        unit:
            The unit the quantity value is specified in.
    """

    value: float
    unit: Unit


def Volt(val: float) -> Quantity:
    """Return a quantity with the unit
    [Volt][laboneq.dsl.calibration.units.Volt].

    Parameters:
        val: The voltage.

    Returns:
        A quantity in volts.
    """
    return Quantity(val, Unit.volt)


def dBm(val: float) -> Quantity:
    """Return a quantity with the unit
    [dBm][laboneq.dsl.calibration.units.dBm].

    Parameters:
        val: The power in dBm.

    Returns:
        A quantity in dBm.
    """
    return Quantity(val, Unit.dBm)
