# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Optional

from attrs import define

from laboneq.compiler.ir.pulse_ir import PrecompClearIR, PulseIR
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.data.compilation_job import SectionSignalPulse


@define(kw_only=True, slots=True)
class PulseSchedule(IntervalSchedule):
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

    def _calculate_timing(
        self,
        schedule_data: ScheduleData,  # type: ignore # noqa: F821
        start: int,
        start_may_change: bool,
    ) -> int:
        # Length must be set via parameter, so nothing to do here
        assert self.length is not None

        if (
            self.is_acquire
            and self.pulse is not None
            and self.pulse.acquire_params is not None
            and self.pulse.acquire_params.handle
        ):
            schedule_data.acquire_pulses.setdefault(
                self.pulse.acquire_params.handle, []
            ).append(self)

        return start

    def __hash__(self):
        return super().__hash__()


@define(kw_only=True, slots=True)
class PrecompClearSchedule(IntervalSchedule):
    pulse: PulseSchedule

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        self.length = 0
        return start

    def to_ir(self):
        return PrecompClearIR(
            children=self.children,
            length=self.length,
            signals=self.signals,
            children_start=self.children_start,
            pulse=PulseIR(
                children=self.pulse.children,
                length=self.pulse.length,
                signals=self.pulse.signals,
                children_start=self.pulse.children_start,
                pulse=self.pulse.pulse,
                amplitude=self.pulse.amplitude,
                amp_param_name=self.pulse.amp_param_name,
                phase=self.pulse.phase,
                offset=self.pulse.offset,
                oscillator_frequency=self.pulse.oscillator_frequency,
                set_oscillator_phase=self.pulse.set_oscillator_phase,
                increment_oscillator_phase=self.pulse.increment_oscillator_phase,
                section=self.pulse.section,
                play_pulse_params=self.pulse.play_pulse_params,
                pulse_pulse_params=self.pulse.pulse_pulse_params,
                is_acquire=self.pulse.is_acquire,
                markers=self.pulse.markers,
            ),
        )

    def __hash__(self):
        return super().__hash__()
