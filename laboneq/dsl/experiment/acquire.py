# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from .pulse import Pulse
from .operation import Operation


@dataclass(init=True, repr=True, order=True)
class Acquire(Operation):
    """Class representing an acquire operation that is used to acquire results."""

    #: Unique identifier of the signal where the result should be acquired.
    signal: str = field(default=None)

    #: Unique identifier of the handle that will be used to access the acquired result.
    handle: str = field(default=None)

    #: Pulse used for the acquisition integration weight (only valid in integration mode).
    kernel: Pulse = field(default=None)

    #: Integration length (only valid in spectroscopy mode).
    length: float = field(default=None)
