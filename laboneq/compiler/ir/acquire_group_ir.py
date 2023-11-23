# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Optional

from attrs import define

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.common.pulse_parameters import encode_pulse_parameters
from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.data.compilation_job import ParameterInfo, SectionSignalPulse


@define(kw_only=True, slots=True)
class AcquireGroupIR(IntervalIR):
    pulses: list[SectionSignalPulse]
    amplitudes: list[float]
    phases: list[float]
    offset: int
    oscillator_frequencies: list[Optional[float]]
    section: str
    play_pulse_params: list[Optional[Dict[str, Any]]]
    pulse_pulse_params: list[Optional[Dict[str, Any]]]

    def generate_event_list(
        self,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> List[Dict]:
        assert self.length is not None
        assert (
            len(self.pulses)
            == len(self.amplitudes)
            == len(self.phases)
            == len(self.oscillator_frequencies)
            == len(self.play_pulse_params)
            == len(self.pulse_pulse_params)
        )

        assert all(
            self.pulses[0].acquire_params.handle == p.acquire_params.handle
            for p in self.pulses
        )
        assert all(
            self.pulses[0].acquire_params.acquisition_type
            == p.acquire_params.acquisition_type
            for p in self.pulses
        )
        start_id = next(id_tracker)
        signal_id = self.pulses[0].signal.uid
        assert all(p.signal.uid == signal_id for p in self.pulses)
        d = {
            "section_name": self.section,
            "signal": signal_id,
            "play_wave_id": [p.pulse.uid for p in self.pulses],
            "parametrized_with": [],
            "phase": self.phases,
            "amplitude": self.amplitudes,
            "chain_element_id": start_id,
            "acquisition_type": [self.pulses[0].acquire_params.acquisition_type],
            "acquire_handle": self.pulses[0].acquire_params.handle,
        }

        if self.oscillator_frequencies is not None:
            d["oscillator_frequency"] = self.oscillator_frequencies

        if self.pulse_pulse_params:
            d["pulse_pulse_parameters"] = [
                encode_pulse_parameters(par) if par is not None else None
                for par in self.pulse_pulse_params
            ]
        if self.play_pulse_params:
            d["play_pulse_parameters"] = [
                encode_pulse_parameters(par) if par is not None else None
                for par in self.play_pulse_params
            ]

        for pulse in self.pulses:
            params_list = [
                getattr(pulse, f).uid
                for f in ("length", "amplitude", "phase", "offset")
                if isinstance(getattr(pulse, f), ParameterInfo)
            ]
            d["parametrized_with"].append(params_list)

        return [
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

    def __hash__(self):
        return super().__hash__()
