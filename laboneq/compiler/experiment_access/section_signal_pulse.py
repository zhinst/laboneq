# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from laboneq.compiler.experiment_access.marker import Marker


@dataclass
class SectionSignalPulse:
    signal_id: str
    pulse_id: Optional[str] = None
    length: Optional[float] = None
    length_param: Optional[str] = None
    amplitude: Optional[float] = None
    amplitude_param: Optional[str] = None
    offset: Optional[float] = None
    offset_param: Optional[str] = None
    phase: Optional[float] = None
    phase_param: Optional[str] = None
    increment_oscillator_phase: Optional[float] = None
    increment_oscillator_phase_param: Optional[str] = None
    set_oscillator_phase: Optional[float] = None
    set_oscillator_phase_param: Optional[str] = None
    acquire_params: Any = None
    play_pulse_parameters: Optional[Any] = None
    pulse_pulse_parameters: Optional[Any] = None
    precompensation_clear: bool = False
    markers: Optional[List[Marker]] = None
