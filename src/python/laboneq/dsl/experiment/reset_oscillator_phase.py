# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .operation import Operation


@classformatter
@attrs.define
class ResetOscillatorPhase(Operation):
    """Operation to reset the phase of the oscillator."""

    #: Unique identifier of the signal whose oscillator phase is reset.
    signal: str | None = attrs.field(default=None)
