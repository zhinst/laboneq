# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from typing import Any

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .operation import Operation
from .pulse import Pulse


@classformatter
@attrs.define
class Acquire(Operation):
    """Class representing an acquire operation that is used to acquire results."""

    #: Unique identifier of the signal where the result should be acquired.
    signal: str | None = attrs.field(default=None)

    #: Unique identifier of the handle that will be used to access the acquired result.
    handle: str | None = attrs.field(default=None)

    #: Pulse(s) used for the acquisition integration weight (only valid in integration mode).
    kernel: Pulse | list[Pulse] | None = attrs.field(default=None)

    #: Integration length (only valid in spectroscopy mode).
    length: float | None = attrs.field(default=None)

    #: Optional (re)binding of user pulse parameters
    pulse_parameters: dict[str, Any] | list[dict[str, Any] | None] | None = attrs.field(
        default=None
    )
