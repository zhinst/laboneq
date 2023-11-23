# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from attrs import define

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.common.play_wave_type import PlayWaveType
from laboneq.compiler.common.pulse_parameters import encode_pulse_parameters
from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.data.compilation_job import ParameterInfo, SectionSignalPulse


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

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        params_list = [
            getattr(self.pulse, f).uid
            for f in ("length", "amplitude", "phase", "offset")
            if isinstance(getattr(self.pulse, f), ParameterInfo)
        ]

        if self.pulse.pulse is not None:
            play_wave_id = self.pulse.pulse.uid
        else:
            play_wave_id = "delay"

        start_id = next(id_tracker)
        d_start = {"id": start_id}
        d_end = {"id": next(id_tracker)}
        d_common = {
            "section_name": self.section,
            "signal": self.pulse.signal.uid,
            "play_wave_id": play_wave_id,
            "parametrized_with": params_list,
            "chain_element_id": start_id,
        }

        if self.amplitude is not None:
            d_start["amplitude"] = self.amplitude

        if self.phase is not None:
            d_start["phase"] = self.phase

        if self.amp_param_name:
            d_start["amplitude_parameter"] = self.amp_param_name

        if self.markers is not None:
            d_start["markers"] = [vars(m) for m in self.markers]

        if self.oscillator_frequency is not None:
            d_start["oscillator_frequency"] = self.oscillator_frequency

        if self.pulse_pulse_params:
            d_start["pulse_pulse_parameters"] = encode_pulse_parameters(
                self.pulse_pulse_params
            )
        if self.play_pulse_params:
            d_start["play_pulse_parameters"] = encode_pulse_parameters(
                self.play_pulse_params
            )

        if self.increment_oscillator_phase is not None:
            d_start["increment_oscillator_phase"] = self.increment_oscillator_phase
        if self.set_oscillator_phase is not None:
            d_start["set_oscillator_phase"] = self.set_oscillator_phase

        is_delay = self.pulse.pulse is None

        if is_delay:
            return [
                {
                    "event_type": EventType.DELAY_START,
                    "time": start,
                    "play_wave_type": PlayWaveType.DELAY.value,
                    **d_start,
                    **d_common,
                },
                {
                    "event_type": EventType.DELAY_END,
                    "time": start + self.length,
                    **d_end,
                    **d_common,
                },
            ]

        if self.is_acquire:
            if self.pulse.acquire_params is not None:
                d_start["acquisition_type"] = [
                    self.pulse.acquire_params.acquisition_type
                ]
                d_start["acquire_handle"] = self.pulse.acquire_params.handle
            else:
                d_start["acquisition_type"] = []
                d_start["acquire_handle"] = None
            return [
                {
                    "event_type": EventType.ACQUIRE_START,
                    "time": start + self.offset,
                    **d_start,
                    **d_common,
                },
                {
                    "event_type": EventType.ACQUIRE_END,
                    "time": start + self.length,
                    **d_end,
                    **d_common,
                },
            ]

        return [
            {
                "event_type": EventType.PLAY_START,
                "time": start + self.offset,
                **d_start,
                **d_common,
            },
            {
                "event_type": EventType.PLAY_END,
                "time": start + self.length,
                **d_end,
                **d_common,
            },
        ]

    def __hash__(self):
        return super().__hash__()


@define(kw_only=True, slots=True)
class PrecompClearIR(IntervalIR):
    pulse: PulseIR

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        return [
            {
                "event_type": EventType.RESET_PRECOMPENSATION_FILTERS,
                "time": start,
                "signal_id": self.pulse.pulse.signal.uid,
                "section_name": self.pulse.section,
                "id": next(id_tracker),
            }
        ]

    def __hash__(self):
        return super().__hash__()
