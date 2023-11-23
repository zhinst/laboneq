# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .operation import Operation
from .pulse import Pulse

if TYPE_CHECKING:
    from .. import Parameter


@classformatter
@dataclass(init=True, repr=True, order=True)
class PlayPulse(Operation):
    """Operation to play a pulse."""

    #: Unique identifier of the signal where the pulse is played.
    signal: str = field(default=None)
    #: Pulse that is played.
    pulse: Pulse = field(default=None)
    #: Amplitude of the pulse.
    amplitude: Union[float, complex, Parameter] = field(default=None)
    #: Increment the phase angle of the modulating oscillator at the start of playing this pulse by this angle (in rad).
    increment_oscillator_phase: Union[float, Parameter] = field(default=None)
    #: Phase of the pulse (in rad).
    phase: float = field(default=None)
    #: Set the phase of the modulating oscillator at the start of playing this pulse to this angle (in rad).
    set_oscillator_phase: float = field(default=None)
    #: Modify the length of the pulse to the given value
    length: Union[float, Parameter] = field(default=None)
    #: Optional (re)binding of user pulse parameters
    pulse_parameters: Optional[Dict[str, Any]] = field(default=None)
    #: Clear the precompensation filter of the signal while playing the pulse.
    precompensation_clear: Optional[bool] = field(default=None)
    #: Instructions for playing marker signals while playing the pulse.
    marker: dict | None = field(default=None)
