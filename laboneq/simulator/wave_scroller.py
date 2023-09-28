# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from enum import Flag
from typing import List, Optional, Set

import numpy as np
from numpy.typing import ArrayLike

from laboneq.simulator.seqc_parser import (
    CommandTableEntryInfo,
    Operation,
    SeqCEvent,
    SeqCSimulation,
)


def _overlaps(a_start: int, a_length: int, b_start: int, b_length: int):
    """Return True if the first and second sample intervals overlap.

    Intervals that touch are considered to overlap.

    Args:
        a_start: The starting sample of the first interval.
        a_length: The number of samples in the first interval.
        b_start: The starting sample of the second interval.
        b_length: The number of samples in the second interval.

    Returns:
        True if the intervals overlap. False otherwise.
    """
    a_end = a_start + a_length
    b_end = b_start + b_length
    return a_start <= b_end and b_start <= a_end


def _slice_copy(
    a: ArrayLike,
    a_start: int,
    b: ArrayLike,
    b_start: int,
    b_len: int,
):
    """Copy up to ``b_len`` samples from array ``b`` to array ``a`` without exceeding
    the end of ``a`` and aligning the two arrays according to their starting samples.

    Args:
        a: An array-like to copy values to.
        a_start: The starting sample of the destination array.
        b: An array-like to copy values from.
        b_start: The starting sample of the source array.
        b_len: The maximum number of samples to copy from the source array.

    Returns:
        Nothing is returned. The destination array is modified inplace.
    """
    a_end = a_start + len(a)
    b_end = b_start + b_len
    if not (a_start < b_end and b_start < a_end):
        # handle the case of intervals that don't overlap or only touch
        return

    left = max(a_start, b_start)
    right = min(a_end, b_end)

    a[left - a_start : right - a_start] = b[left - b_start : right - b_start]


def _slice_add(
    a: ArrayLike,
    a_start: int,
    b: ArrayLike,
    b_start: int,
    b_len: int,
):
    """Add up to ``b_len`` samples from array ``b`` to the corresponding entries from
    array ``a`` without exceeding the end of ``a`` and aligning the two arrays according
    to their starting samples.

    Args:
        a: An array-like to add values to.
        a_start: The starting sample of the destination array.
        b: An array-like to add values from.
        b_start: The starting sample of the source array.
        b_len: The maximum number of samples to copy from the source array.

    Returns:
        Nothing is returned. The destination array is modified inplace.
    """
    a_end = a_start + len(a)
    b_end = b_start + b_len
    if not (a_start < b_end and b_start < a_end):
        # handle the case of intervals that don't overlap or only touch
        return

    left = max(a_start, b_start)
    right = min(a_end, b_end)

    a[left - a_start : right - a_start] += b[left - b_start : right - b_start]


def _slice_set(a: ArrayLike, a_start: int, b, b_start: int, b_len: int):
    """Set up to ``b_len`` samples in array ``a`` to value ``b`` without exceeding
    the end of ``a`` and aligning the intervals according to their starting samples.

    Args:
        a: An array like to copy values to.
        a_start: The starting sample of the destination array.
        b: The value to set.
        b_start: The starting sample of the source event.
        b_len: The length of the source event.

    Returns:
        Nothing is returned. The destination array is modified inplace.
    """
    a_end = a_start + len(a)
    b_end = b_start + b_len
    if not (a_start < b_end and b_start < a_end):
        # handle the case of intervals that don't overlap or only touch
        return

    left = max(a_start, b_start)
    right = min(a_end, b_end)

    a[left - a_start : right - a_start] = b


class SimTarget(Flag):
    NONE = 0
    PLAY = 1
    ACQUIRE = 2
    TRIGGER = 4
    FREQUENCY = 8
    MARKER = 16


class WaveScroller:
    def __init__(
        self,
        *,
        ch: List[int],
        sim_targets: SimTarget,
        sim: SeqCSimulation,
    ):
        self.ch = ch
        self.sim = sim

        self.is_shfqa = sim.device_type == "SHFQA"
        self.is_spectroscopy = sim.is_spectroscopy

        self.sim_targets = sim_targets

        self.wave_snippet = None
        self.marker_snippet = None
        self.acquire_snippet = None
        self.trigger_snippet = None
        self.frequency_snippet = None
        self.time_axis = None

        self.last_trig_set_samples = 0
        self.last_trig = 0
        self.last_freq_set_samples = 0
        self.last_freq = np.nan
        self.last_played_value = 0
        self.oscillator_phase: Optional[float] = None

        # Only Hirzel supports 4 registers, for other devices, the simulator simply does
        # not use the extra registers.
        self.amplitude_registers = [1.0 for _ in range(4)]

        self.processors = {
            Operation.PLAY_WAVE: self._process_play_wave,
            Operation.PLAY_HOLD: self._process_play_hold,
            Operation.PLAY_ZERO: self._process_play_zero,
            Operation.START_QA: self._process_start_qa,
            Operation.SET_TRIGGER: self._process_set_trigger,
            Operation.SET_OSC_FREQ: self._process_set_osc_freq,
        }

    def is_output(self) -> bool:
        return any(
            t in self.sim_targets
            for t in [
                SimTarget.PLAY,
                SimTarget.TRIGGER,
                SimTarget.FREQUENCY,
                SimTarget.MARKER,
            ]
        )

    def prepare(self, snippet_start_samples: int, snippet_length: int):
        if SimTarget.PLAY in self.sim_targets:
            if len(self.ch) > 1 or self.is_shfqa:
                self.wave_snippet = np.zeros(snippet_length, dtype=np.complex128)
            else:
                self.wave_snippet = np.zeros(snippet_length, dtype=np.float64)
        if SimTarget.ACQUIRE in self.sim_targets:
            self.acquire_snippet = np.zeros(snippet_length, dtype=np.uint8)
        if SimTarget.TRIGGER in self.sim_targets:
            self.trigger_snippet = np.zeros(snippet_length, dtype=np.uint8)
        if SimTarget.MARKER in self.sim_targets:
            self.marker_snippet = np.zeros(snippet_length, dtype=np.complex128)
        if SimTarget.FREQUENCY in self.sim_targets:
            self.frequency_snippet = np.full(snippet_length, np.nan, dtype=np.float64)

    def target_ops(self) -> Set[Operation]:
        target_ops: Set[Operation] = set()
        if (
            SimTarget.ACQUIRE in self.sim_targets
            or SimTarget.TRIGGER in self.sim_targets
            or SimTarget.PLAY in self.sim_targets
            and self.is_shfqa
        ):
            target_ops.add(Operation.START_QA)
        if SimTarget.PLAY in self.sim_targets and not self.is_shfqa:
            target_ops.add(Operation.PLAY_WAVE)
            target_ops.add(Operation.PLAY_HOLD)
            target_ops.add(Operation.PLAY_ZERO)
        if SimTarget.TRIGGER in self.sim_targets:
            target_ops.add(Operation.SET_TRIGGER)
        if SimTarget.FREQUENCY in self.sim_targets:
            target_ops.add(Operation.SET_OSC_FREQ)
        return target_ops

    def _process_play_wave(self, event: SeqCEvent, snippet_start_samples: int):
        wave_data_indices = event.args[0]
        ct_info: CommandTableEntryInfo = event.args[1]

        wave = None

        if len(self.ch) > 1:
            ct_abs_phase = ct_info.abs_phase if ct_info is not None else None
            ct_rel_phase = ct_info.rel_phase if ct_info is not None else None
            if ct_abs_phase is not None:
                self.oscillator_phase = ct_abs_phase / (180 / math.pi)
            if ct_rel_phase is not None:
                self.oscillator_phase = (
                    self.oscillator_phase or 0.0
                ) + ct_rel_phase / (180 / math.pi)

            if wave_data_indices is not None:
                wave = 1j * self.sim.waves[wave_data_indices[self.ch[1] % 2]]
                wave += self.sim.waves[wave_data_indices[self.ch[0] % 2]]

                # If the command table phase is set, assume that the signal is complex
                # (rather than 2x real)
                if self.oscillator_phase is not None:
                    wave *= np.exp(-1j * self.oscillator_phase)

        else:
            if wave_data_indices is not None:
                wave_data_index = wave_data_indices[self.ch[0] % 2]
                if wave_data_index is None:
                    # the requested channel is the first half of a pair of real channels,
                    # but it is unused.
                    return
                wave = self.sim.waves[wave_data_index]
                wave = (
                    wave.copy()
                )  # so amplitude below does not mutate original waveform
                # Note: CT phase not implemented on RF signals

        if ct_info is not None:
            amp_register = ct_info.amp_register or 0
            if (abs_amp := ct_info.abs_amplitude) is not None:
                self.amplitude_registers[amp_register] = abs_amp
            elif (rel_amp := ct_info.rel_amplitude) is not None:
                self.amplitude_registers[amp_register] += rel_amp
            if wave is not None:
                wave *= self.amplitude_registers[amp_register]

        if wave is not None:
            _slice_copy(
                self.wave_snippet,
                snippet_start_samples,
                wave,
                event.start_samples,
                event.length_samples,
            )
            self.last_played_value = wave[event.length_samples - 1]

        markers = event.args[2] if len(event.args) > 2 else {}

        if markers.get("marker1"):
            _slice_copy(
                self.marker_snippet,
                snippet_start_samples,
                self.sim.waves[event.args[0][2]],
                event.start_samples,
                event.length_samples,
            )

        if markers.get("marker2"):
            wave_arg_pos = 3 if event.args[2]["marker1"] else 2
            _slice_add(
                self.marker_snippet,
                snippet_start_samples,
                1j * self.sim.waves[event.args[0][wave_arg_pos]],
                event.start_samples,
                event.length_samples,
            )

    def _process_play_hold(self, event: SeqCEvent, snippet_start_samples: int):
        _slice_set(
            self.wave_snippet,
            snippet_start_samples,
            self.last_played_value,
            event.start_samples,
            event.length_samples,
        )

    def _process_play_zero(self, event: SeqCEvent, snippet_start_samples: int):
        _slice_set(
            self.wave_snippet,
            snippet_start_samples,
            0.0,
            event.start_samples,
            event.length_samples,
        )
        self.last_played_value = 0.0

    def _process_start_qa(self, event: SeqCEvent, snippet_start_samples: int):
        if SimTarget.PLAY in self.sim_targets and self.is_shfqa:
            self._process_shfqa_gen(event, snippet_start_samples)
        if (
            SimTarget.ACQUIRE in self.sim_targets
            or SimTarget.TRIGGER in self.sim_targets
        ):
            self._process_acquire(event, snippet_start_samples)

    def _process_shfqa_gen(self, event: SeqCEvent, snippet_start_samples: int):
        generator_mask: int = event.args[0]
        if self.ch[0] < 0:
            # The old_output_simulator sets ChannelInfo("QAResult", -1, SimTarget.ACQUIRE) to
            # skip producing the SHFQA acquire play pulse, so we support that here too.
            return

        def retrieve_wave(real_idx, imag_idx):
            wave = 1j * self.sim.waves[imag_idx]
            wave += self.sim.waves[real_idx]
            return wave

        wave_indices = event.args[4]

        if self.is_spectroscopy:
            spectroscopy_mask = 0
            assert generator_mask == spectroscopy_mask
            wave = retrieve_wave(
                wave_indices[spectroscopy_mask][0], wave_indices[spectroscopy_mask][1]
            )
            _slice_copy(
                self.wave_snippet,
                snippet_start_samples,
                wave,
                event.start_samples,
                event.length_samples,
            )
            self.last_played_value = wave[event.length_samples - 1]
        else:
            wave_iter = 0
            wave = None
            for gen_index in range(16):
                if (generator_mask & (1 << gen_index)) != 0:
                    if wave is None:
                        wave = retrieve_wave(
                            wave_indices[wave_iter][0], wave_indices[wave_iter][1]
                        )
                    else:
                        wave += retrieve_wave(
                            wave_indices[wave_iter][0], wave_indices[wave_iter][1]
                        )
                    wave_iter = wave_iter + 1
            _slice_copy(
                self.wave_snippet,
                snippet_start_samples,
                wave,
                event.start_samples,
                event.length_samples,
            )
            self.last_played_value = wave[event.length_samples - 1]

    def _process_acquire(self, event: SeqCEvent, snippet_start_samples: int):
        if SimTarget.ACQUIRE in self.sim_targets:
            integrator_mask: int = event.args[1]
            test_mask = 0xFFFF if self.ch[0] < 0 else (1 << self.ch[0])
            if (integrator_mask & test_mask) != 0:
                measurement_delay_samples: int = event.args[2]
                wave_start_samples = (
                    event.start_samples
                    - snippet_start_samples
                    + measurement_delay_samples
                )
                if 0 <= wave_start_samples < len(self.acquire_snippet):
                    self.acquire_snippet[wave_start_samples] = 1
        if SimTarget.TRIGGER in self.sim_targets:
            trigger_index = 5 if self.is_shfqa else 4
            if event.args[trigger_index] is None:
                self._process_set_trigger(
                    SeqCEvent(event.start_samples, 0, Operation.SET_TRIGGER, [0]),
                    snippet_start_samples,
                )
            else:
                self._process_set_trigger(
                    SeqCEvent(
                        event.start_samples,
                        0,
                        Operation.SET_TRIGGER,
                        [event.args[trigger_index]],
                    ),
                    snippet_start_samples,
                )

    def _process_set_trigger(self, event: SeqCEvent, snippet_start_samples: int):
        value: int = int(event.args[0])
        wave_start_samples = event.start_samples - snippet_start_samples
        if 0 <= wave_start_samples <= len(self.trigger_snippet):
            self.trigger_snippet[
                self.last_trig_set_samples : wave_start_samples
            ] = self.last_trig
        self.last_trig_set_samples = max(0, wave_start_samples)
        self.last_trig = value

    def _process_set_osc_freq(self, event: SeqCEvent, snippet_start_samples: int):
        oscillator: int = event.args[0]
        # TODO(2K): Track oscillator switching, currently osc 0 is hard-coded for
        #           Hw sweeps
        if oscillator == 0:
            frequency: float = event.args[1]
            wave_start_samples = event.start_samples - snippet_start_samples
            if 0 <= wave_start_samples <= len(self.frequency_snippet):
                self.frequency_snippet[
                    self.last_freq_set_samples : wave_start_samples
                ] = self.last_freq
            self.last_freq_set_samples = max(0, wave_start_samples)
            self.last_freq = frequency

    def process(self, event: SeqCEvent, snippet_start_samples: int):
        processor = self.processors.get(event.operation)
        if processor is not None:
            processor(event, snippet_start_samples)

    def finalize(self):
        if self.trigger_snippet is not None:
            if self.last_trig_set_samples < len(self.trigger_snippet):
                self.trigger_snippet[self.last_trig_set_samples :] = self.last_trig

        if self.frequency_snippet is not None:
            if self.last_freq_set_samples < len(self.frequency_snippet):
                self.frequency_snippet[self.last_freq_set_samples :] = self.last_freq

    def calc_snippet(
        self,
        start_secs: float,
        length_secs: float,
    ):
        time_delay_secs = self.sim.output_port_delay if self.is_output() else 0.0
        time_delay_secs += self.sim.startup_delay
        start_samples = int(
            np.round((start_secs - time_delay_secs) * self.sim.sampling_rate)
        )
        length_samples = int(np.round(length_secs * self.sim.sampling_rate))
        if start_samples < 0:
            # truncate any part of the interval that extends into negative
            # sample counts (there are no events with negative samples)
            length_samples = max(0, length_samples + start_samples)
            start_samples = 0
        end_samples = start_samples + length_samples

        # filter relevant events into events pre-interval and events that
        # overlap the interval, keeping only the last of each kind of
        # operation
        pre_events = {}
        interval_events = []
        target_ops = self.target_ops()
        for ev in self.sim.events:
            if ev.start_samples > end_samples:
                break
            if ev.operation not in target_ops:
                continue
            if _overlaps(
                start_samples,
                length_samples,
                ev.start_samples,
                ev.length_samples,
            ):
                interval_events.append(ev)
            else:
                pre_events[ev.operation] = ev
        pre_events = sorted(pre_events.values(), key=lambda ev: ev.start_samples)

        # truncate sample length to the end of the last contained event
        if interval_events:
            ev_end = max(
                ev.start_samples + (ev.length_samples or 1) for ev in interval_events
            )
            end_samples = min(ev_end, end_samples)
            length_samples = end_samples - start_samples
        else:
            end_samples = start_samples
            length_samples = 0

        # prepare and populate the snippets
        self.prepare(start_samples, length_samples)
        for ev in pre_events:
            self.process(ev, start_samples)
        for ev in interval_events:
            self.process(ev, start_samples)
        self.finalize()

        exact_start_secs = start_samples / self.sim.sampling_rate + time_delay_secs
        exact_length_secs = (length_samples - 1) / self.sim.sampling_rate
        self.time_axis = np.linspace(
            exact_start_secs, exact_start_secs + exact_length_secs, length_samples
        )
