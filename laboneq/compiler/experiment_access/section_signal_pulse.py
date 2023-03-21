# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

from laboneq.compiler.experiment_access.marker import Marker


@dataclass
class SectionSignalPulse:
    pulse_id: str
    length: float
    length_param: str
    amplitude: float
    amplitude_param: str
    play_mode: str
    signal_id: str
    offset: float
    offset_param: str
    acquire_params: Any
    phase: float
    phase_param: str
    increment_oscillator_phase: float
    increment_oscillator_phase_param: str
    set_oscillator_phase: float
    set_oscillator_phase_param: str
    play_pulse_parameters: Optional[Any]
    pulse_pulse_parameters: Optional[Any]
    precompensation_clear: bool
    markers: Optional[List[Marker]]
