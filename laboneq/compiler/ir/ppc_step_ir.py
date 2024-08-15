# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


from attrs import define

from laboneq.compiler.ir.interval_ir import IntervalIR


@define(kw_only=True, slots=True)
class PPCStepIR(IntervalIR):
    section: str
    qa_device: str
    qa_channel: int
    ppc_device: str
    ppc_channel: int
    trigger_duration: int

    pump_power: float | None = None
    pump_frequency: float | None = None
    probe_power: float | None = None
    probe_frequency: float | None = None
    cancellation_phase: float | None = None
    cancellation_attenuation: float | None = None
