# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from typing import TYPE_CHECKING, Optional

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.experiment.operation import Operation

if TYPE_CHECKING:
    from laboneq.dsl import Parameter


@classformatter
@attrs.define
class Delay(Operation):
    """Class representing a delay operation for a specific signal."""

    #: Unique identifier of the signal where the delay should be applied.
    signal: str | None = attrs.field(default=None)

    #:  Duration of the delay.
    time: float | Parameter | None = attrs.field(default=None)

    #: Clear the precompensation filter of the signal.
    precompensation_clear: Optional[bool] = attrs.field(default=None)
