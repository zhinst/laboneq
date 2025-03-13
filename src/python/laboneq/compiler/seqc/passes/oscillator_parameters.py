# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections import defaultdict
import sortedcollections
import math

from laboneq.compiler import ir

from typing import KeysView
from laboneq.compiler.common.compiler_settings import TINYSAMPLE
from laboneq.compiler.common.device_type import DeviceType


class SoftwareOscillatorParameters:
    """Software oscillator parameters with timing information.

    This class should be an temporary solution while the pulse oscillator
    information lives in the event list.
    """

    def __init__(
        self,
        frequencies: dict[str, tuple[int, float]],
        phases: dict[str, dict[int, float]] | None = None,
    ):
        self._freq_by_signal: dict[str, sortedcollections.NearestDict] = {}
        for k, v in frequencies.items():
            self._freq_by_signal[k] = sortedcollections.NearestDict(
                {time: freq for time, freq in sorted(v, key=lambda x: x[0])},
                rounding=sortedcollections.NearestDict.NEAREST_PREV,
            )
        # Precompute minimum, as otherwise non-existed (too early) timestamp would raise an KeyError
        self._mins = {sig: vals.keys()[0] for sig, vals in self._freq_by_signal.items()}
        self._phase_by_signal = phases or {}

    def freq_keys(self) -> KeysView:
        return self._freq_by_signal.keys()

    def freq_at(self, identifier: str, time: float) -> float | None:
        """Oscillator frequency at given timestamp for given identifier."""
        if time < self._mins[identifier]:
            return None
        return self._freq_by_signal[identifier][time]

    def phase_at(self, identifier: str, time: float) -> float | None:
        """Oscillator phase at given timestamp for given identifier."""
        if identifier not in self._phase_by_signal:
            return None
        return self._phase_by_signal[identifier][time]


class _PhaseTracker:
    def __init__(self):
        # Cumulative phase per signal
        self._cumulative = {}
        # Reference time of last phase set time per signal
        self._reference_times = {}
        self._global_reset_time = 0

    def set(self, signal: str, time: int, value: float):
        self._cumulative[signal] = value
        self._reference_times[signal] = time

    def increment(self, signal: str, value: float):
        if signal not in self._cumulative:
            self._cumulative[signal] = 0.0
        self._cumulative[signal] += value

    def global_reset(self, time: int):
        # Reset all signal phases set/incremented so far
        self._global_reset_time = time
        for signal in self._cumulative:
            self._cumulative[signal] = 0.0

    def phase_now(self, signal: str) -> tuple[float, int]:
        return self._cumulative.get(signal, 0.0), max(
            self._reference_times.get(signal, 0), self._global_reset_time
        )


class _PickOscillatorParameters:
    """Traverse the IR nodes and find software oscillator parameters."""

    def __init__(self):
        self._sw_osc_freqs = defaultdict(list)
        self._sw_osc_phases = {}
        self._phase_tracker = _PhaseTracker()

    def run(self, node: ir.RootScheduleIR) -> tuple[dict, dict]:
        self.visit(node, 0)
        result = self._sw_osc_freqs, self._sw_osc_phases
        self._sw_osc_freqs = defaultdict(list)
        self._sw_osc_phases = {}
        self._phase_tracker = _PhaseTracker()
        return result

    def visit(self, node: ir.IntervalIR, start: int):
        visitor = getattr(self, f"visit_{node.__class__.__name__}", self.generic_visit)
        return visitor(node, start)

    def generic_visit(self, node: ir.IntervalIR, start: int) -> None:
        for start_ch, child in node.iter_children():
            # Absolute times
            self.visit(child, start + start_ch)

    def visit_SetOscillatorFrequencyIR(
        self, node: ir.SetOscillatorFrequencyIR, start: int
    ) -> None:
        for osc, value in zip(node.oscillators, node.values):
            if osc.is_hardware:
                continue
            for sig in osc.signals:
                self._sw_osc_freqs[sig].append((start, value))

    def visit_InitialOscillatorFrequencyIR(
        self, node: ir.InitialOscillatorFrequencyIR, start: int
    ) -> None:
        # Initial frequency is only in the RootSchedule.
        for osc, value in zip(node.oscillators, node.values):
            if osc.is_hardware:
                continue
            for sig in osc.signals:
                self._sw_osc_freqs[sig].append((start, value))

    def visit_PhaseResetIR(self, node: ir.PhaseResetIR, start: int):
        if not node.reset_sw_oscillators:
            return
        self._phase_tracker.global_reset(start)

    def visit_PulseIR(self, node: ir.PulseIR, start: int):
        if node.pulse.signal.oscillator and node.pulse.signal.oscillator.is_hardware:
            assert not node.set_oscillator_phase, "Cannot set phase of HW oscillators"
            return
        if node.is_acquire:
            return
        [signal] = node.signals
        # "Delay" is without pulse
        timing = start + node.offset if node.pulse.pulse else start
        device_type = DeviceType.from_device_info_type(
            node.pulse.signal.device.device_type
        )
        if device_type.is_qa_device:
            self._sw_osc_phases.setdefault(signal, {})[timing] = 0.0
            return
        # Set oscillator priority over incrementing
        if node.set_oscillator_phase is not None:
            self._phase_tracker.set(signal, start, node.set_oscillator_phase)
        elif node.increment_oscillator_phase is not None:
            self._phase_tracker.increment(signal, node.increment_oscillator_phase)
        phase_now, ref_time = self._phase_tracker.phase_now(signal)
        # Use last set SW oscillator frequency value
        if signal in self._sw_osc_freqs:
            oscillator_frequency = self._sw_osc_freqs[signal][-1][-1]
        else:
            oscillator_frequency = 0.0
        t = (timing - ref_time) * TINYSAMPLE
        phase = t * 2.0 * math.pi * oscillator_frequency + phase_now
        self._sw_osc_phases.setdefault(signal, {})[timing] = phase


def calculate_oscillator_parameters(
    node: ir.IntervalIR,
) -> SoftwareOscillatorParameters:
    """Analysis pass for finding software oscillator parameters at specific times."""
    return SoftwareOscillatorParameters(*_PickOscillatorParameters().run(node))
