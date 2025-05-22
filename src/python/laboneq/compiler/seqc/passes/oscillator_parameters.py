# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections import defaultdict
import sortedcollections

from laboneq.compiler import ir

from typing import KeysView


class SoftwareOscillatorParameters:
    """Software oscillator parameters with timing information.

    This class should be an temporary solution while the pulse oscillator
    information lives in the event list.
    """

    def __init__(
        self,
        frequencies: dict[str, tuple[int, float]],
    ):
        self._freq_by_signal: dict[str, sortedcollections.NearestDict] = {}
        for k, v in frequencies.items():
            self._freq_by_signal[k] = sortedcollections.NearestDict(
                {time: freq for time, freq in sorted(v, key=lambda x: x[0])},
                rounding=sortedcollections.NearestDict.NEAREST_PREV,
            )
        # Precompute minimum, as otherwise non-existed (too early) timestamp would raise an KeyError
        self._mins = {sig: vals.keys()[0] for sig, vals in self._freq_by_signal.items()}

    def freq_keys(self) -> KeysView:
        return self._freq_by_signal.keys()

    def freq_at(self, identifier: str, time: float) -> float | None:
        """Oscillator frequency at given timestamp for given identifier."""
        if time < self._mins[identifier]:
            return None
        return self._freq_by_signal[identifier][time]


class _PickOscillatorParameters:
    """Traverse the IR nodes and find software oscillator parameters."""

    def __init__(self):
        self._sw_osc_freqs = defaultdict(list)

    def run(self, node: ir.RootScheduleIR) -> tuple[dict, dict]:
        self.visit(node, 0)
        result = self._sw_osc_freqs
        self._sw_osc_freqs = defaultdict(list)
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


def calculate_oscillator_parameters(
    node: ir.IntervalIR,
) -> SoftwareOscillatorParameters:
    """Analysis pass for finding software oscillator parameters at specific times."""
    return SoftwareOscillatorParameters(_PickOscillatorParameters().run(node))
