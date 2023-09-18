# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

import numpy as np
from attrs import define

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.common.play_wave_type import PlayWaveType
from laboneq.compiler.common.pulse_parameters import encode_pulse_parameters
from laboneq.compiler.scheduler.interval_schedule import IntervalSchedule
from laboneq.data.compilation_job import ParameterInfo, SectionSignalPulse


@define(kw_only=True, slots=True)
class PulseSchedule(IntervalSchedule):
    pulse: SectionSignalPulse
    amplitude: float
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

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        assert self.absolute_start is not None
        params_list = []
        for f in ("length", "amplitude", "phase", "offset"):
            if isinstance(getattr(self.pulse, f), ParameterInfo):
                params_list.append(getattr(self.pulse, f).uid)

        if self.pulse.pulse is not None:
            play_wave_id = self.pulse.pulse.uid
        else:
            play_wave_id = "delay"

        amplitude_resolution = pow(
            2, getattr(settings, "AMPLITUDE_RESOLUTION_BITS", 24)
        )
        amplitude = (
            np.round(self.amplitude * amplitude_resolution) / amplitude_resolution
        )

        start_id = next(id_tracker)
        d = {
            "section_name": self.section,
            "signal": self.pulse.signal.uid,
            "play_wave_id": play_wave_id,
            "parametrized_with": params_list,
            "phase": self.phase,
            "amplitude": amplitude,
            "chain_element_id": start_id,
        }

        if self.markers is not None:
            d["markers"] = [vars(m) for m in self.markers]

        if self.oscillator_frequency is not None:
            d["oscillator_frequency"] = self.oscillator_frequency

        if self.pulse_pulse_params:
            d["pulse_pulse_parameters"] = encode_pulse_parameters(
                self.pulse_pulse_params
            )
        if self.play_pulse_params:
            d["play_pulse_parameters"] = encode_pulse_parameters(self.play_pulse_params)

        osc_events = []
        osc_common = {
            "time": start,
            "section_name": self.section,
            "signal": self.pulse.signal.uid,
        }
        if self.increment_oscillator_phase:
            osc_events.append(
                {
                    "event_type": EventType.INCREMENT_OSCILLATOR_PHASE,
                    "increment_oscillator_phase": self.increment_oscillator_phase,
                    "id": next(id_tracker),
                    **osc_common,
                }
            )
        if self.set_oscillator_phase is not None:
            osc_events.append(
                {
                    "event_type": EventType.SET_OSCILLATOR_PHASE,
                    "set_oscillator_phase": self.set_oscillator_phase,
                    "id": next(id_tracker),
                    **osc_common,
                }
            )

        is_delay = self.pulse.pulse is None

        if is_delay:
            return osc_events + [
                {
                    "event_type": EventType.DELAY_START,
                    "time": start,
                    "id": start_id,
                    "play_wave_type": PlayWaveType.DELAY.value,
                    **d,
                },
                {
                    "event_type": EventType.DELAY_END,
                    "time": start + self.length,
                    "id": next(id_tracker),
                    **d,
                },
            ]

        if self.is_acquire:
            if self.pulse.acquire_params is not None:
                d["acquisition_type"] = [self.pulse.acquire_params.acquisition_type]
                d["acquire_handle"] = self.pulse.acquire_params.handle
            else:
                d["acquisition_type"] = []
                d["acquire_handle"] = None
            return osc_events + [
                {
                    "event_type": EventType.ACQUIRE_START,
                    "time": start + self.offset,
                    "id": start_id,
                    **d,
                },
                {
                    "event_type": EventType.ACQUIRE_END,
                    "time": start + self.length,
                    "id": next(id_tracker),
                    **d,
                },
            ]

        return osc_events + [
            {
                "event_type": EventType.PLAY_START,
                "time": start + self.offset,
                "id": start_id,
                **d,
            },
            {
                "event_type": EventType.PLAY_END,
                "time": start + self.length,
                "id": next(id_tracker),
                **d,
            },
        ]

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

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        assert self.absolute_start is not None
        return [
            {
                "event_type": EventType.RESET_PRECOMPENSATION_FILTERS,
                "time": start,
                "signal_id": self.pulse.pulse.signal.uid,
                "section_name": self.pulse.section,
                "id": next(id_tracker),
            }
        ]

    def _calculate_timing(self, _schedule_data, start: int, *__, **___) -> int:
        self.length = 0
        return start

    def __hash__(self):
        super().__hash__()
