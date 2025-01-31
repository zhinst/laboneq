# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict

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
    set_oscillator_phase: float | None = None
    increment_oscillator_phase: float | None = None
    incr_phase_param_name: str | None = None
    section: str
    play_pulse_params: Dict[str, Any] | None = None
    pulse_pulse_params: Dict[str, Any] | None = None
    is_acquire: bool
    markers: Any = None
    integration_length: int | None = None


@define(kw_only=True, slots=True)
class PrecompClearIR(IntervalIR):
    """Precompensation clear command on a specific signal."""
