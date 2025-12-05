# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.data.compilation_job import PulseDef


@define(kw_only=True, slots=True)
class AcquireGroupIR(IntervalIR):
    pulses: list[PulseDef]
    amplitudes: list[float]
    play_pulse_params: list[Optional[Dict[str, Any]]]
    pulse_pulse_params: list[Optional[Dict[str, Any]]]
    handle: str
    acquisition_type: str
