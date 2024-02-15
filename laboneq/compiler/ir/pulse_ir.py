# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.data.compilation_job import SectionSignalPulse


@define(kw_only=True, slots=True)
class PulseIR(IntervalIR):
    pulse: SectionSignalPulse
    amplitude: float
    amp_param_name: str | None = None
    phase: float
    offset: int
    oscillator_frequency: Optional[float] = None
    set_oscillator_phase: Optional[float] = None
    increment_oscillator_phase: Optional[float] = None
    section: str
    play_pulse_params: Optional[Dict[str, Any]] = None
    pulse_pulse_params: Optional[Dict[str, Any]] = None
    is_acquire: bool
    markers: Any = None


@define(kw_only=True, slots=True)
class PrecompClearIR(IntervalIR):
    pulse: PulseIR
