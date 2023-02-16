# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.experiment_access.experiment_dao import SectionSignalPulse
from laboneq.compiler.new_scheduler.interval_schedule import IntervalSchedule


@dataclass(frozen=True)
class PulseSchedule(IntervalSchedule):
    pulse: SectionSignalPulse
    amplitude: float
    phase: float
    offset: float
    oscillator_frequency: Optional[float]
    set_oscillator_phase: Optional[float]
    increment_oscillator_phase: Optional[float]
    section: str

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops=False,
        settings: CompilerSettings = None,
    ) -> List[Dict]:
        params_list = []
        for field in ("length_param", "amplitude_param", "phase_param", "offset_param"):
            if getattr(self.pulse, field) is not None:
                params_list.append(getattr(self.pulse, field))

        play_wave_id = self.pulse.pulse_id or "delay"

        amplitude_resolution = pow(2, settings.AMPLITUDE_RESOLUTION_BITS)
        amplitude = round(self.amplitude * amplitude_resolution) / amplitude_resolution

        start_id = next(id_tracker)
        d = {
            "section_name": self.section,
            "signal": self.pulse.signal_id,
            "play_wave_id": play_wave_id,
            "parametrized_with": params_list,
            "phase": self.phase,
            "amplitude": amplitude,
            "chain_element_id": start_id
            # todo: play_wave_type, pulse parameters
        }

        if self.oscillator_frequency is not None:
            d["oscillator_frequency"] = self.oscillator_frequency

        osc_events = []
        osc_common = {
            "time": start,
            "section_name": self.section,
            "signal": self.pulse.signal_id,
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
        if self.set_oscillator_phase:
            osc_events.append(
                {
                    "event_type": EventType.SET_OSCILLATOR_PHASE,
                    "set_oscillator_phase": self.set_oscillator_phase,
                    "id": next(id_tracker),
                    **osc_common,
                }
            )

        is_delay = self.pulse.pulse_id is None

        if is_delay:
            return osc_events + [
                {
                    "event_type": EventType.DELAY_START,
                    "time": start,
                    "id": start_id,
                    **d,
                },
                {
                    "event_type": EventType.DELAY_END,
                    "time": start + self.length,
                    "id": next(id_tracker),
                    **d,
                },
            ]

        is_acquire = self.is_acquire()
        if is_acquire:
            d.update(
                {
                    "acquisition_type": [self.pulse.acquire_params.acquisition_type],
                    "acquire_handle": self.pulse.acquire_params.handle,
                }
            )
            return osc_events + [
                {
                    "event_type": EventType.ACQUIRE_START,
                    "time": start,
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
            {"event_type": EventType.PLAY_START, "time": start, "id": start_id, **d},
            {
                "event_type": EventType.PLAY_END,
                "time": start + self.length,
                "id": next(id_tracker),
                **d,
            },
        ]

    def is_acquire(self):
        return self.pulse.acquire_params is not None

    def __hash__(self):
        return super().__hash__()
