# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Union

from laboneq.dsl.experiment.operation import Operation

if TYPE_CHECKING:
    from laboneq.dsl import Parameter


@dataclass(init=True, repr=True, order=True)
class Delay(Operation):
    """Class representing a delay operation for a specific signal."""

    #: Unique identifier of the signal where the delay should be applied.
    signal: str = field(default=None)

    #:  Duration of the delay.
    time: Union[float, Parameter] = field(default=None)

    #: Clear the precompensation filter of the signal.
    precompensation_clear: Optional[bool] = field(default=None)
