# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

from laboneq.compiler import ir
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.compiler_settings import TINYSAMPLE
from laboneq.compiler.common.integration_times import (
    IntegrationTimes,
)
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.seqc.feedback_register_allocator import FeedbackRegisterAllocator
from laboneq.compiler.seqc.measurement_calculator import (
    MeasurementCalculator,
    SignalDelays,
    _IntermediateSignalIntegrationInfo,
    _MeasurementInfo,
    calculate_integration_times_from_intermediate_infos,
)


@dataclass
class MeasurementInfo:
    integration_times: IntegrationTimes
    signal_delays: SignalDelays
    qa_signals_by_handle: dict[str, SignalObj]
    simultaneous_acquires: list[dict[str, str | None]]
    feedback_register_allocator: FeedbackRegisterAllocator


class AcquisitionInfo(NamedTuple):
    start: int
    signal: str
    handle: str | None


def is_delay(node: ir.PulseIR) -> bool:
    return node.pulse.pulse is None


class CollectAcquiringAwgs:
    def __init__(self, signals: dict[str, SignalObj]):
        super().__init__()
        self.signals = signals

        self.awgs_with_acquires: set[AwgKey] = set()

    def visit(self, node: ir.IntervalIR, start: int = 0):
        visitor = getattr(self, f"visit_{node.__class__.__name__}", self.generic_visit)
        visitor(node, start)

    def generic_visit(self, node: ir.IntervalIR, start: int):
        for child_start, child in node.iter_children():
            if not getattr(child, "shadow", False):
                self.visit(child, start + child_start)

    def visit_PulseIR(self, node: ir.PulseIR, start: int):
        if node.is_acquire and not is_delay(node):
            signal = node.pulse.signal.uid

            # MeasurementCalculator
            if (signal_info := self.signals.get(signal, None)) is not None:
                self.awgs_with_acquires.add(signal_info.awg.key)

    def visit_AcquireGroupIR(self, node: ir.AcquireGroupIR, start: int):
        if node.pulses:
            signal = node.pulses[0].signal.uid
            # MeasurementCalculator
            if (signal_info := self.signals.get(signal, None)) is not None:
                self.awgs_with_acquires.add(signal_info.awg.key)


class CollectInfo:
    def __init__(
        self,
        signals: dict[str, SignalObj],
        awgs_with_acquires: set[AwgKey],
    ) -> None:
        super().__init__()
        self.signals: dict[str, SignalObj] = signals
        self.awgs_with_acquires: set[AwgKey] = awgs_with_acquires

        self.section_start_time: int = 0
        self.shadow_levels: int = 0

        self.qa_signals_by_handle: dict[str, SignalObj] = {}
        self.open_acquisitions_stack: list[list[AcquisitionInfo]] = [[]]
        self.feedback_register_allocator = FeedbackRegisterAllocator(signals)

        # map from (section_uid, awg_id) to _MeasurementInfo
        self.measurement_infos: dict[tuple[str, AwgKey], _MeasurementInfo] = {}
        # map from (section_uid, signal_id) tuples to _IntermediateSignalIntegrationInfo
        self.intermediate_signal_infos: dict[
            tuple[str, str], _IntermediateSignalIntegrationInfo
        ] = {}
        self.first_event_on_signal_in_section = True

    def visit(self, node: ir.IntervalIR, start: int = 0):
        visitor = getattr(self, f"visit_{node.__class__.__name__}", self.generic_visit)
        visitor(node, start)

    def generic_visit(self, node: ir.IntervalIR, start: int):
        previous_section_start_time = self.section_start_time
        self.section_start_time = start

        for child_start, child in node.iter_children():
            is_shadow = getattr(child, "shadow", False)
            if is_shadow:
                self.shadow_levels += 1
            self.visit(child, start + child_start)
            if is_shadow:
                self.shadow_levels -= 1
        self.section_start_time = previous_section_start_time

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
        signal = node.pulse.signal.uid
        if not is_delay(node) and signal in self.signals:
            params = node.pulse.acquire_params
            handle = params.handle if params else None

            # generate_acquire_map
            if node.is_acquire:
                start_time = start + node.offset
                self.open_acquisitions_stack[-1].append(
                    AcquisitionInfo(start_time, signal, handle)
                )

            if self.shadow_levels == 0:
                pulse_start = start + node.offset
                length = node.integration_length if node.is_acquire else node.length
                assert length is not None
                end = start + length
                self._extract_measurement_infos(
                    signal, node.is_acquire, pulse_start, end, node.section, handle
                )

    def visit_AcquireGroupIR(self, node: ir.AcquireGroupIR, start: int):
        if not node.pulses:
            return
        signal = node.pulses[0].signal.uid
        if signal in self.signals:
            params = node.pulses[0].acquire_params
            handle = params.handle if params else None

            # generate_acquire_map
            start_time = start + node.offset
            self.open_acquisitions_stack[-1].append(
                AcquisitionInfo(start_time, signal, handle)
            )

            # MeasurementCalculator
            if self.shadow_levels == 0:
                pulse_start = start + node.offset
                assert node.length is not None
                end = start + node.length
                self._extract_measurement_infos(
                    node.pulses[0].signal.uid,
                    True,
                    pulse_start,
                    end,
                    node.section,
                    handle,
                )

    def _extract_measurement_infos(
        self,
        signal_id: str,
        is_acquire: bool,
        start_ts: int,
        end_ts: int,
        section: str,
        handle: str | None,
    ):
        signal_info = self.signals[signal_id]

        if is_acquire:
            # qa_signals_by_handle
            if handle is not None:
                if handle not in self.qa_signals_by_handle:
                    self.qa_signals_by_handle[handle] = signal_info
                else:
                    assert self.qa_signals_by_handle[handle] == signal_info

        # MeasurementCalculator
        awg_id = signal_info.awg.key
        if awg_id not in self.awgs_with_acquires:
            return
        if (section, signal_id) not in self.intermediate_signal_infos:
            inter_info = _IntermediateSignalIntegrationInfo()
            self.intermediate_signal_infos[(section, signal_id)] = inter_info
            self.first_event_on_signal_in_section = True
        else:
            inter_info = self.intermediate_signal_infos[(section, signal_id)]
            self.first_event_on_signal_in_section = False

        if (section, awg_id) not in self.measurement_infos:
            measurement_info = _MeasurementInfo(
                device_type=signal_info.awg.device_type,
                section_uid=section,
                section_start=self.section_start_time * TINYSAMPLE,
            )
            self.measurement_infos[(section, awg_id)] = measurement_info
        else:
            measurement_info = self.measurement_infos[(section, awg_id)]

        if not self.first_event_on_signal_in_section:
            operation = "acquire" if is_acquire else "play"
            raise ValueError(
                f"There are multiple {operation} operations in section {section!r} on signal {signal_id!r}."
                f" A section with acquire signals may only contain a single {operation} operation per signal."
            )

        start = start_ts * TINYSAMPLE
        end = end_ts * TINYSAMPLE
        inter_info.start = start + signal_info.delay_signal
        inter_info.end = end + signal_info.delay_signal

        if is_acquire:
            inter_info.is_play = False
            if measurement_info.acquire_start is None:
                measurement_info.acquire_start = inter_info.start
            else:
                if measurement_info.acquire_start != inter_info.start:
                    raise ValueError(
                        f"There are multiple acquire start times in section {section!r}."
                        f" In a section with an acquire, all acquire signals must start at the same time."
                        f" Signal {signal_id!r} starts at {inter_info.start}."
                        f" This conflicts with the signals {measurement_info.acquire_signals} that start at"
                        f" {measurement_info.acquire_start}."
                    )
            measurement_info.acquire_signals.append(signal_id)
            if measurement_info.acquire_end is None:
                measurement_info.acquire_end = inter_info.end
            else:
                measurement_info.acquire_end = max(
                    measurement_info.acquire_end, inter_info.end
                )
        else:
            inter_info.is_play = True
            if measurement_info.play_start is None:
                measurement_info.play_start = inter_info.start
            else:
                if measurement_info.play_start != inter_info.start:
                    raise ValueError(
                        f"There are multiple play start times in section {section!r}."
                        f" In a section with an acquire, all play signals must start at the same time."
                        f" Signal {signal_id!r} starts at {inter_info.start}."
                        f" This conflicts with the signals {measurement_info.play_signals} that start at"
                        f" {measurement_info.play_start}."
                    )
            measurement_info.play_signals.append(signal_id)
            if measurement_info.play_end is None:
                measurement_info.play_end = inter_info.end
            else:
                measurement_info.play_end = max(
                    measurement_info.play_end, inter_info.end
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
    first_pass = CollectAcquiringAwgs(signals)
    first_pass.visit(root)
    awgs_with_acquires = first_pass.awgs_with_acquires

    data_collector = CollectInfo(
        signals,
        awgs_with_acquires=awgs_with_acquires,
    )
    data_collector.visit(root)

    intermediate_signal_infos = data_collector.intermediate_signal_infos
    measurement_infos = data_collector.measurement_infos

    integration_times = calculate_integration_times_from_intermediate_infos(
        signals, intermediate_signal_infos
    )
    signal_delays = MeasurementCalculator.calculate_signal_delays(
        measurement_infos, signals
    )

    # generate_acquire_map
    simultaneous_acquires: dict[int, dict[str, str | None]] = {}
    [acquisitions] = data_collector.open_acquisitions_stack
    for acq in acquisitions:
        time_events = simultaneous_acquires.setdefault(acq.start, {})
        time_events[acq.signal] = acq.handle

    return MeasurementInfo(
        integration_times=integration_times,
        signal_delays=signal_delays,
        simultaneous_acquires=list(simultaneous_acquires.values()),
        qa_signals_by_handle=data_collector.qa_signals_by_handle,
        feedback_register_allocator=data_collector.feedback_register_allocator,
    )
