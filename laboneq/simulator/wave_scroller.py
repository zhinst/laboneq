# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Flag
from typing import List, Set, Tuple

import numpy as np
from laboneq.simulator.seqc_parser import Operation, SeqCEvent, SeqCSimulation
from numpy import typing as npt


class SimTarget(Flag):
    NONE = 0
    PLAY = 1
    ACQUIRE = 2
    TRIGGER = 4
    FREQUENCY = 8


class WaveScroller:
    def __init__(
        self,
        *,
        ch: List[int],
        sim_targets: SimTarget,
        sim: SeqCSimulation,
    ):
        self.ch = ch
        self.sim_targets = sim_targets
        self.sim = sim

        self.is_shfqa = sim.device_type == "SHFQA"

        self.wave_snippet = None
        self.acquire_snippet = None
        self.trigger_snippet = None
        self.frequency_snippet = None
        self.time_axis = None

        self.last_trig_set_samples = 0
        self.last_trig = 0
        self.last_freq_set_samples = 0
        self.last_freq = None

        self.processors = {
            Operation.PLAY_WAVE: self._process_play_wave,
            Operation.START_QA: self._process_start_qa,
            Operation.SET_TRIGGER: self._process_set_trigger,
            Operation.SET_OSC_FREQ: self._process_set_osc_freq,
        }

    def is_output(self) -> bool:
        return any(
            t in self.sim_targets
            for t in [SimTarget.PLAY, SimTarget.TRIGGER, SimTarget.FREQUENCY]
        )

    def prepare(self, snippet_length: int):
        if SimTarget.PLAY in self.sim_targets:
            if len(self.ch) > 1 or self.is_shfqa:
                self.wave_snippet = np.zeros(snippet_length, dtype=np.complex128)
            else:
                self.wave_snippet = np.zeros(snippet_length, dtype=np.float64)
        if SimTarget.ACQUIRE in self.sim_targets:
            self.acquire_snippet = np.zeros(snippet_length, dtype=np.uint8)
        if SimTarget.TRIGGER in self.sim_targets:
            self.trigger_snippet = np.zeros(snippet_length, dtype=np.uint8)
        if SimTarget.FREQUENCY in self.sim_targets:
            self.frequency_snippet = np.zeros(snippet_length, dtype=np.float64)

    def target_events(self) -> Set[Operation]:
        target_events: Set[Operation] = set()
        if (
            SimTarget.ACQUIRE in self.sim_targets
            or SimTarget.PLAY in self.sim_targets
            and self.is_shfqa
        ):
            target_events.add(Operation.START_QA)
        if SimTarget.PLAY in self.sim_targets and not self.is_shfqa:
            target_events.add(Operation.PLAY_WAVE)
        if SimTarget.TRIGGER in self.sim_targets:
            target_events.add(Operation.SET_TRIGGER)
        if SimTarget.FREQUENCY in self.sim_targets:
            target_events.add(Operation.SET_OSC_FREQ)
        return target_events

    def _process_play_wave(self, event: SeqCEvent, snippet_start_samples: int):
        if len(self.ch) > 1:
            wave = 1j * self.sim.waves[event.args[0][self.ch[1] % 2]]
            wave += self.sim.waves[event.args[0][self.ch[0] % 2]]
        else:
            wave = self.sim.waves[event.args[0][self.ch[0] % 2]]
        wave_start_samples = event.start_samples - snippet_start_samples
        self.wave_snippet[wave_start_samples : wave_start_samples + len(wave)] = wave

    def _process_start_qa(self, event: SeqCEvent, snippet_start_samples: int):
        if SimTarget.PLAY in self.sim_targets and self.is_shfqa:
            self._process_shfqa_gen(event, snippet_start_samples)
        if SimTarget.ACQUIRE in self.sim_targets:
            self._process_acquire(event, snippet_start_samples)

    def _process_shfqa_gen(self, event: SeqCEvent, snippet_start_samples: int):
        generator_mask: int = event.args[0]
        if (generator_mask & (1 << self.ch[0])) != 0:
            wave_start_samples = event.start_samples - snippet_start_samples
            wave = 1j * self.sim.waves[event.args[4][1]]
            wave += self.sim.waves[event.args[4][0]]
            # TODO(2K): ensure wave doesn't exceed snippet boundary,
            # as the wave length is not included in the event length
            self.wave_snippet[
                wave_start_samples : wave_start_samples + len(wave)
            ] = wave

    def _process_acquire(self, event: SeqCEvent, snippet_start_samples: int):
        integrator_mask: int = event.args[1]
        test_mask = 0xFFFF if self.ch[0] < 0 else (1 << self.ch[0])
        if (integrator_mask & test_mask) != 0:
            measurement_delay_samples: int = event.args[2]
            wave_start_samples = (
                event.start_samples - snippet_start_samples + measurement_delay_samples
            )
            if wave_start_samples < len(self.acquire_snippet):
                self.acquire_snippet[wave_start_samples] = 1.0

    def _process_set_trigger(self, event: SeqCEvent, snippet_start_samples: int):
        value: int = int(event.args[0])
        wave_start_samples = event.start_samples - snippet_start_samples
        if (
            wave_start_samples <= len(self.trigger_snippet)
            and self.last_trig is not None
        ):
            self.trigger_snippet[
                self.last_trig_set_samples : wave_start_samples
            ] = self.last_trig
        self.last_trig_set_samples = wave_start_samples
        self.last_trig = value

    def _process_set_osc_freq(self, event: SeqCEvent, snippet_start_samples: int):
        oscillator: int = event.args[0]
        # TODO(2K): Track oscillator switching, currently osc 0 is hard-coded for Hw sweeps
        if oscillator == 0:
            frequency: float = event.args[1]
            wave_start_samples = event.start_samples - snippet_start_samples
            if (
                wave_start_samples <= len(self.frequency_snippet)
                and self.last_freq is not None
            ):
                self.frequency_snippet[
                    self.last_freq_set_samples : wave_start_samples
                ] = self.last_freq
            self.last_freq_set_samples = wave_start_samples
            self.last_freq = frequency

    def process(self, event: SeqCEvent, snippet_start_samples: int):
        processor = self.processors.get(event.operation)
        if processor is not None:
            processor(event, snippet_start_samples)

    def trim(self, offset: int, length: int):
        if self.wave_snippet is not None:
            self.wave_snippet = self.wave_snippet[offset : offset + length]
        if self.acquire_snippet is not None:
            self.acquire_snippet = self.acquire_snippet[offset : offset + length]
        if self.trigger_snippet is not None:
            self.trigger_snippet = self.trigger_snippet[offset : offset + length]
            if self.last_trig_set_samples < len(self.trigger_snippet):
                self.trigger_snippet[self.last_trig_set_samples :] = self.last_trig
        if self.frequency_snippet is not None:
            self.frequency_snippet = self.frequency_snippet[offset : offset + length]
            if self.last_freq_set_samples < len(self.frequency_snippet):
                self.frequency_snippet[self.last_freq_set_samples :] = self.last_freq

    def calc_snippet(
        self,
        start_secs: float,
        length_secs: float,
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        time_delay_secs = self.sim.output_port_delay if self.is_output() else 0.0
        time_delay_secs += self.sim.startup_delay
        target_events = self.target_events()
        start_samples = int(
            np.round((start_secs - time_delay_secs) * self.sim.sampling_rate)
        )
        length_samples = int(np.round(length_secs * self.sim.sampling_rate))
        end_samples = start_samples + length_samples
        start_ev_idx = next(
            (
                i
                for i, p in enumerate(self.sim.events)
                if p.start_samples > start_samples
            ),
            None,
        )
        if start_ev_idx is None:
            start_ev_idx = len(self.sim.events) - 1
        elif start_ev_idx > 0:
            start_ev_idx -= 1

        snippet_start_samples = self.sim.events[start_ev_idx].start_samples
        while (
            start_ev_idx > 0
            and self.sim.events[start_ev_idx - 1].start_samples == snippet_start_samples
        ):
            start_ev_idx -= 1
        ev_idx = start_ev_idx
        cur_time_samples = snippet_start_samples
        op_events: List[SeqCEvent] = []
        while ev_idx < len(self.sim.events) and cur_time_samples < end_samples:
            ev = self.sim.events[ev_idx]
            if ev.operation in target_events:
                op_events.append(ev)
            cur_time_samples += ev.length_samples
            ev_idx += 1

        self.prepare(cur_time_samples - snippet_start_samples)
        for ev in op_events:
            self.process(ev, snippet_start_samples)

        # clip to actually available samples, even if wider range requested
        end_samples = min(end_samples, cur_time_samples)
        start_samples = max(0, start_samples)
        length_samples = end_samples - start_samples
        if length_samples <= 0:
            return np.array([]), np.array([])

        ofs = start_samples - snippet_start_samples
        self.trim(ofs, length_samples)

        exact_start_secs = start_samples / self.sim.sampling_rate + time_delay_secs
        exact_length_secs = (length_samples - 1) / self.sim.sampling_rate
        self.time_axis = np.linspace(
            exact_start_secs, exact_start_secs + exact_length_secs, length_samples
        )
