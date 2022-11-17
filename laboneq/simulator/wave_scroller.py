# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import numpy as np
from laboneq.simulator.seqc_parser import Operation, SeqCEvent, SeqCSimulation
from numpy import typing as npt


class WaveScroller:
    def __init__(self, simulations: Dict[str, SeqCSimulation]):
        self.simulations = simulations
        self.repo = {}
        self.ev_repo = {}

    def _calc_delay(self, sim: SeqCSimulation, is_output: bool) -> float:
        time_delay_secs = sim.output_port_delay if is_output else 0.0
        time_delay_secs += sim.startup_delay
        return time_delay_secs

    def _get_snippet(
        self,
        sim: SeqCSimulation,
        start_secs: float,
        length_secs: float,
        time_delay_secs: float,
        target_events: List[Operation],
        processor: Callable[[SeqCEvent, npt.ArrayLike, int], None],
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        start_samples = int(
            np.round((start_secs - time_delay_secs) * sim.sampling_rate)
        )
        length_samples = int(np.round(length_secs * sim.sampling_rate))
        end_samples = start_samples + length_samples
        start_ev_idx = next(
            (i for i, p in enumerate(sim.events) if p.start_samples > start_samples),
            None,
        )
        if start_ev_idx is None:
            start_ev_idx = len(sim.events) - 1
        elif start_ev_idx > 0:
            start_ev_idx -= 1

        snippet_start_samples = sim.events[start_ev_idx].start_samples
        ev_idx = start_ev_idx
        cur_time_samples = snippet_start_samples
        op_events: List[SeqCEvent] = []
        while ev_idx < len(sim.events) and cur_time_samples < end_samples:
            ev = sim.events[ev_idx]
            if ev.operation in target_events:
                op_events.append(ev)
            cur_time_samples += ev.length_samples
            ev_idx += 1

        snippet = np.zeros(
            cur_time_samples - snippet_start_samples, dtype=np.complex128
        )
        for ev in op_events:
            processor(ev, snippet, snippet_start_samples)

        # clip to actually available samples, even if wider range requested
        end_samples = min(end_samples, cur_time_samples)
        start_samples = max(0, start_samples)
        length_samples = end_samples - start_samples
        if length_samples <= 0:
            return np.array([]), np.array([])

        exact_start_secs = start_samples / sim.sampling_rate + time_delay_secs
        exact_length_secs = (length_samples - 1) / sim.sampling_rate
        time_axis = np.linspace(
            exact_start_secs, exact_start_secs + exact_length_secs, length_samples
        )
        ofs = start_samples - snippet_start_samples
        samples = snippet[ofs : ofs + length_samples]
        return time_axis, samples

    def get_play_snippet(
        self,
        start_secs: float,
        length_secs: float,
        prog: str,
        ch: int,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        sim = self.simulations[prog]
        time_delay_secs = self._calc_delay(sim, is_output=True)
        if sim.device_type == "SHFQA":

            def process_startQA_event(
                ev: SeqCEvent, snippet: npt.ArrayLike, snippet_start_samples: int
            ):
                generator_mask: int = ev.args[0]
                if (generator_mask & (1 << ch)) != 0:
                    wave_start_samples = ev.start_samples - snippet_start_samples
                    wave = 1j * sim.waves[ev.args[4][1]]
                    wave += sim.waves[ev.args[4][0]]
                    # TODO(2K): ensure wave doesn't exceed snippet boundary,
                    # as the wave length is not included in the event length
                    snippet[wave_start_samples : wave_start_samples + len(wave)] = wave

            return self._get_snippet(
                sim,
                start_secs,
                length_secs,
                time_delay_secs,
                [Operation.START_QA],
                process_startQA_event,
            )
        else:

            def process_playWave_event(
                ev: SeqCEvent, snippet: npt.ArrayLike, snippet_start_samples: int
            ):
                wave = sim.waves[ev.args[0][ch]]
                wave_start_samples = ev.start_samples - snippet_start_samples
                snippet[wave_start_samples : wave_start_samples + len(wave)] = wave

            return self._get_snippet(
                sim,
                start_secs,
                length_secs,
                time_delay_secs,
                [Operation.PLAY_WAVE],
                process_playWave_event,
            )

    def get_acquire_snippet(
        self,
        start_secs: float,
        length_secs: float,
        prog: str,
        ch: int,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        sim = self.simulations[prog]
        time_delay_secs = self._calc_delay(sim, is_output=False)

        def process_startQA_event(
            ev: SeqCEvent, snippet: npt.ArrayLike, snippet_start_samples: int
        ):
            integrator_mask: int = ev.args[1]
            if (integrator_mask & ch) != 0:
                measurement_delay_samples: int = ev.args[2]
                wave_start_samples = (
                    ev.start_samples - snippet_start_samples + measurement_delay_samples
                )
                if wave_start_samples < len(snippet):
                    snippet[wave_start_samples] = 1.0

        return self._get_snippet(
            sim,
            start_secs,
            length_secs,
            time_delay_secs,
            [Operation.START_QA],
            process_startQA_event,
        )

    def get_freq_snippet(
        self,
        start_secs: float,
        length_secs: float,
        prog: str,
        ch: int,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        sim = self.simulations[prog]
        time_delay_secs = self._calc_delay(sim, is_output=False)
        state = SimpleNamespace(last_freq_set_samples=0, last_freq=None)

        def process_setOscFreq_event(
            ev: SeqCEvent, snippet: npt.ArrayLike, snippet_start_samples: int
        ):
            oscillator: int = ev.args[0]
            if oscillator == ch:
                frequency: float = ev.args[1]
                wave_start_samples = ev.start_samples - snippet_start_samples
                if wave_start_samples <= len(snippet) and state.last_freq is not None:
                    snippet[
                        state.last_freq_set_samples : wave_start_samples
                    ] = state.last_freq
                state.last_freq_set_samples = wave_start_samples
                state.last_freq = frequency

        time_ax, snippet = self._get_snippet(
            sim,
            start_secs,
            length_secs,
            time_delay_secs,
            [Operation.SET_OSC_FREQ],
            process_setOscFreq_event,
        )
        if state.last_freq_set_samples < len(snippet):
            snippet[state.last_freq_set_samples :] = state.last_freq
        return time_ax, snippet

    def get_trigger_snippet(
        self,
        start_secs: float,
        length_secs: float,
        prog: str,
        ch: int,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        sim = self.simulations[prog]
        time_delay_secs = self._calc_delay(sim, is_output=False)
        state = SimpleNamespace(last_trig_set_samples=0, last_trig=0)

        def process_setTrigger_event(
            ev: SeqCEvent, snippet: npt.ArrayLike, snippet_start_samples: int
        ):
            value: int = int(ev.args[0])
            wave_start_samples = ev.start_samples - snippet_start_samples
            if wave_start_samples <= len(snippet) and state.last_trig is not None:
                snippet[
                    state.last_trig_set_samples : wave_start_samples
                ] = state.last_trig
            state.last_trig_set_samples = wave_start_samples
            state.last_trig = value

        time_ax, snippet = self._get_snippet(
            sim,
            start_secs,
            length_secs,
            time_delay_secs,
            [Operation.SET_TRIGGER],
            process_setTrigger_event,
        )
        if state.last_trig_set_samples < len(snippet):
            snippet[state.last_trig_set_samples :] = state.last_trig
        return time_ax, snippet

    def get_non_zero(self, prog: str):
        if prog in self.ev_repo:
            return self.ev_repo[prog]
        sim = self.simulations[prog]
        time_delay_secs = self._calc_delay(sim, is_output=True)
        l = len(sim.events)
        x = np.empty(l)

        total = 0
        for ev in sim.events:
            if ev.operation != Operation.PLAY_ZERO:
                x[total] = ev.start_samples / sim.sampling_rate + time_delay_secs
                total += 1
        ev_list = x[:total]
        self.ev_repo[prog] = ev_list
        return ev_list
