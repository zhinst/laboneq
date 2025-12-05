# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict

from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.data.compilation_job import PulseDef


@define(kw_only=True, slots=True)
class PulseIR(IntervalIR):
    pulse: PulseDef | None
    amplitude: float
    amp_param_name: str | None = None
    phase: float
    set_oscillator_phase: float | None = None
    increment_oscillator_phase: float | None = None
    incr_phase_param_name: str | None = None
    play_pulse_params: Dict[str, Any] | None = None
    pulse_pulse_params: Dict[str, Any] | None = None
    pulse_params_id: int | None = None
    markers: Any = None
    # Acquisition related fields
    is_acquire: bool
    integration_length: int | None = None
    handle: str | None = None
    acquisition_type: str | None = None


@define(kw_only=True, slots=True)
class PrecompClearIR(IntervalIR):
    """Precompensation clear command on a specific signal."""
