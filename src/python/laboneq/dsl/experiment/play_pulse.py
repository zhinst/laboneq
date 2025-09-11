# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
import numpy as np
from typing import TYPE_CHECKING, Any, Optional
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.core.exceptions import LabOneQException

from .operation import Operation
from .pulse import Pulse

if TYPE_CHECKING:
    from .. import Parameter


def _validate_marker_keys(_instance, _attribute, value):
    """Validate that marker dictionary contains only allowed keys."""
    if value is not None:
        valid_marker_keys = {"marker1", "marker2"}
        invalid_keys = value.keys() - valid_marker_keys
        if invalid_keys:
            raise LabOneQException(
                f"Invalid marker keys: {sorted(invalid_keys)}. "
                "Only 'marker1' and 'marker2' are supported."
            )


@classformatter
@attrs.define
class PlayPulse(Operation):
    """Operation to play a pulse."""

    #: Unique identifier of the signal where the pulse is played.
    signal: str | None = attrs.field(default=None)
    #: Pulse that is played.
    pulse: Pulse | None = attrs.field(default=None)
    #: Amplitude of the pulse.
    amplitude: float | complex | np.number | Parameter | None = attrs.field(
        default=None
    )
    #: Increment the phase angle of the modulating oscillator at the start of playing this pulse by this angle (in rad).
    increment_oscillator_phase: float | Parameter | None = attrs.field(default=None)
    #: Phase of the pulse (in rad).
    phase: float | np.number | Parameter | None = attrs.field(default=None)
    #: Set the phase of the modulating oscillator at the start of playing this pulse to this angle (in rad).
    set_oscillator_phase: float | None = attrs.field(default=None)
    #: Modify the length of the pulse to the given value
    length: float | Parameter | None = attrs.field(default=None)
    #: Optional (re)binding of user pulse parameters
    pulse_parameters: Optional[dict[str, Any]] = attrs.field(default=None)
    #: Clear the precompensation filter of the signal while playing the pulse.
    precompensation_clear: Optional[bool] = attrs.field(default=None)
    #: Instructions for playing marker signals while playing the pulse.
    marker: dict | None = attrs.field(default=None, validator=_validate_marker_keys)
