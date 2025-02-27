# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from laboneq.compiler import ir
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.seqc.feedback_register_allocator import FeedbackRegisterAllocator


@dataclass
class MeasurementInfo:
    qa_signals_by_handle: dict[str, SignalObj]
    simultaneous_acquires: list[dict[str, str | None]]
    feedback_register_allocator: FeedbackRegisterAllocator


class AcquisitionInfo(NamedTuple):
    start: int
    signal: str
    handle: str | None


def is_delay(node: ir.PulseIR) -> bool:
    return node.pulse.pulse is None


class CollectInfo:
    def __init__(self, signals: dict[str, SignalObj]):
        super().__init__()
        self.signals = signals

        self.qa_signals_by_handle: dict[str, SignalObj] = {}
        self.open_acquisitions_stack: list[list[AcquisitionInfo]] = [[]]
        self.feedback_register_allocator = FeedbackRegisterAllocator(signals)

    def visit(self, node: ir.IntervalIR, start: int = 0):
        visitor = getattr(self, f"visit_{node.__class__.__name__}", self.generic_visit)
        return visitor(node, start)

    def generic_visit(self, node: ir.IntervalIR, start: int):
        for child_start, child in node.iter_children():
            self.visit(child, start + child_start)

    def visit_LoopIR(self, node: ir.LoopIR, start: int):
        # generate_acquire_map
        loop_iteration = next(
            (c for c in node.children if type(c) is ir.LoopIterationIR), None
        )
        is_prng_loop = (
            loop_iteration is not None and loop_iteration.prng_sample is not None
        )
        if is_prng_loop:
            self.open_acquisitions_stack.append([])

        self.generic_visit(node, start)

        if is_prng_loop:
            open_acquisitions = self.open_acquisitions_stack.pop()
            length = (node.length or 0) / node.iterations
            unrolled_acquires: list[AcquisitionInfo] = []
            for original_acquire in open_acquisitions:
                unrolled_acquires.extend(
                    [
                        AcquisitionInfo(
                            original_acquire.start + round(i * length),
                            original_acquire.signal,
                            original_acquire.handle,
                        )
                        for i in range(node.iterations)
                    ]
                )
            self.open_acquisitions_stack[-1].extend(unrolled_acquires)

    def visit_PulseIR(self, node: ir.PulseIR, start: int):
        if node.is_acquire and not is_delay(node):
            signal = node.pulse.signal.uid
            params = node.pulse.acquire_params
            handle = params.handle if params else None

            # qa_signals_by_handle
            if handle is not None:
                signal_obj = self.signals[signal]
                if handle not in self.qa_signals_by_handle:
                    self.qa_signals_by_handle[handle] = signal_obj
                else:
                    assert self.qa_signals_by_handle[handle] == signal_obj

            # generate_acquire_map
            start_time = start + node.offset
            self.open_acquisitions_stack[-1].append(
                AcquisitionInfo(start_time, signal, handle)
            )

    def visit_AcquireGroupIR(self, node: ir.AcquireGroupIR, start: int):
        if node.pulses:
            signal = node.pulses[0].signal.uid
            params = node.pulses[0].acquire_params
            handle = params.handle if params else None

            # qa_signals_by_handle
            if handle is not None:
                signal_obj = self.signals[signal]
                if handle not in self.qa_signals_by_handle:
                    self.qa_signals_by_handle[handle] = signal_obj
                else:
                    assert self.qa_signals_by_handle[handle] == signal_obj

            # generate_acquire_map
            start_time = start + node.offset
            self.open_acquisitions_stack[-1].append(
                AcquisitionInfo(start_time, signal, handle)
            )

    def visit_MatchIR(self, node: ir.MatchIR, start: int):
        # feedback_register_allocator
        if node.handle is not None:
            self.feedback_register_allocator.set_feedback_path(
                node.handle, not node.local
            )
        self.generic_visit(node, start)


def collect_measurement_info(
    root: ir.IntervalIR, signals: dict[str, SignalObj]
) -> MeasurementInfo:
    data_collector = CollectInfo(signals)
    data_collector.visit(root)

    # generate_acquire_map
    simultaneous_acquires: dict[int, dict[str, str | None]] = {}
    [acquisitions] = data_collector.open_acquisitions_stack
    for acq in acquisitions:
        time_events = simultaneous_acquires.setdefault(acq.start, {})
        time_events[acq.signal] = acq.handle

    return MeasurementInfo(
        simultaneous_acquires=list(simultaneous_acquires.values()),
        qa_signals_by_handle=data_collector.qa_signals_by_handle,
        feedback_register_allocator=data_collector.feedback_register_allocator,
    )
