# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import Any, Callable, Iterator
from laboneq.compiler.common.compiler_settings import TINYSAMPLE

from laboneq.compiler import ir as ir_mod
from laboneq.compiler.common.play_wave_type import PlayWaveType
from laboneq.compiler.event_list.event_type import EventList, EventType
from laboneq.data.compilation_job import ParameterInfo


class EventListGenerator:
    """Event list generator."""

    def __init__(
        self,
        id_tracker: Iterator[int] | None = None,
        expand_loops: bool = False,
    ):
        self.id_tracker = itertools.count() if id_tracker is None else id_tracker
        self.expand_loops = expand_loops

    # This is consumed mainly by the PSV and attached to the
    # CompiledExperiment with compiler flag `OUTPUT_EXTRAS`
    def run(
        self, ir: ir_mod.IntervalIR, start: int, max_events: int | float = float("inf")
    ) -> EventList:
        return self.visit(ir, start, max_events)

    def visit(
        self, ir: ir_mod.IntervalIR, start: int, max_events: int | float = float("inf")
    ) -> EventList:
        visitor: Callable[..., EventList] = getattr(
            self, f"visit_{ir.__class__.__name__}", self.generic_visit
        )
        return visitor(ir, start, max_events)

    def generic_visit(
        self, node: ir_mod.IntervalIR, start: int, max_events: int | float
    ):
        return [
            event
            for child in node.children
            for event in self.visit(child, start, max_events)
        ]

    def visit_SectionIR(
        self, section_ir: ir_mod.SectionIR, start: int, max_events: int | float
    ) -> EventList:
        assert section_ir.length is not None

        # We'll wrap the child events in the section start and end events
        max_events -= 2

        children_events = self.generate_children_events(section_ir, start, max_events)

        start_id = next(self.id_tracker)
        d = {"section_name": section_ir.section, "chain_element_id": start_id}
        return [
            {
                "event_type": EventType.SECTION_START,
                "time": start,
                "id": start_id,
                **d,
            },
            *[e for l in children_events for e in l],
            {
                "event_type": EventType.SECTION_END,
                "time": start + section_ir.length,
                "id": next(self.id_tracker),
                **d,
            },
        ]

    def visit_AcquireGroupIR(
        self,
        acquire_group_ir: ir_mod.AcquireGroupIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert acquire_group_ir.length is not None
        assert (
            len(acquire_group_ir.pulses)
            == len(acquire_group_ir.amplitudes)
            == len(acquire_group_ir.phases)
            == len(acquire_group_ir.play_pulse_params)
            == len(acquire_group_ir.pulse_pulse_params)
        )

        assert all(
            acquire_group_ir.pulses[0].acquire_params.handle == p.acquire_params.handle
            for p in acquire_group_ir.pulses
        )
        start_id = next(self.id_tracker)
        signal_id = acquire_group_ir.pulses[0].signal.uid
        assert all(p.signal.uid == signal_id for p in acquire_group_ir.pulses)
        assert all(p.pulse is not None for p in acquire_group_ir.pulses)
        d = {
            "section_name": acquire_group_ir.section,
            "signal": signal_id,
            "play_wave_id": [p.pulse.uid for p in acquire_group_ir.pulses],
            "parametrized_with": [],
            "phase": acquire_group_ir.phases,
            "chain_element_id": start_id,
            "acquire_handle": acquire_group_ir.pulses[0].acquire_params.handle,
        }
        d["id_pulse_params"] = acquire_group_ir.pulse_pulse_params
        for pulse in acquire_group_ir.pulses:
            params_list = [
                getattr(pulse, f).uid
                for f in ("length", "amplitude", "phase")
                if isinstance(getattr(pulse, f), ParameterInfo)
            ]
            d["parametrized_with"].append(params_list)
        return [
            {
                "event_type": EventType.ACQUIRE_START,
                "time": start,
                "id": start_id,
                **d,
            },
            {
                "event_type": EventType.ACQUIRE_END,
                "time": start + acquire_group_ir.length,
                "id": next(self.id_tracker),
                **d,
            },
        ]

    def visit_SetOscillatorFrequencyIR(
        self,
        ir: ir_mod.SetOscillatorFrequencyIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert ir.length is not None
        retval = []
        for param, osc, value in zip(ir.params, ir.oscillators, ir.values):
            if not osc.is_hardware:
                continue
            start_id = next(self.id_tracker)
            retval.extend(
                [
                    {
                        "event_type": EventType.SET_OSCILLATOR_FREQUENCY_START,
                        "time": start,
                        "parameter": {"id": param},
                        "iteration": ir.iteration,
                        "value": value,
                        "section_name": ir.section,
                        "device_id": osc.device,
                        "signal": osc.signals,
                        "oscillator_id": osc.id,
                        "id": start_id,
                        "chain_element_id": start_id,
                    },
                ]
            )
        return retval

    def visit_InitialOscillatorFrequencyIR(
        self,
        ir: ir_mod.InitialOscillatorFrequencyIR,
        start: int,
        max_events: int,
    ) -> EventList:
        return []

    def visit_RootScheduleIR(
        self,
        root_ir: ir_mod.RootScheduleIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert root_ir.length is not None
        children_events = self.generate_children_events(root_ir, start, max_events - 2)

        return [e for l in children_events for e in l]

    def visit_LoopIR(
        self,
        loop_ir: ir_mod.LoopIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert loop_ir.children_start is not None
        assert loop_ir.length is not None

        # We'll later wrap the child events in some extra events, see below.
        max_events -= 3

        if not loop_ir.compressed:  # unrolled loop
            children_events = self.generate_children_events(
                loop_ir,
                start,
                max_events,
                subsection_events=False,
            )
        else:
            children_events = [
                self.visit(
                    loop_ir.children[0],
                    start + loop_ir.children_start[0],
                    max_events,
                )
            ]
            iteration_event = children_events[0][-1]
            assert iteration_event["event_type"] == EventType.LOOP_ITERATION_END
            iteration_event["compressed"] = True
            if self.expand_loops:
                [prototype] = loop_ir.children
                assert prototype.length is not None
                assert isinstance(prototype, ir_mod.LoopIterationIR)
                iteration_start = start
                for iteration in range(1, loop_ir.iterations):
                    max_events -= len(children_events[-1])
                    if max_events <= 0:
                        break
                    iteration_start += prototype.length
                    shadow_iteration = prototype.compressed_iteration(iteration)
                    children_events.append(
                        self.visit(
                            shadow_iteration,
                            iteration_start,
                            max_events,
                        )
                    )

        for child_list in children_events:
            for e in child_list:
                if "nesting_level" in e:
                    # todo: pass the current level as an argument, rather than
                    #  incrementing after the fact
                    e["nesting_level"] += 1

        start_id = next(self.id_tracker)
        d = {
            "section_name": loop_ir.section,
            "nesting_level": 0,
            "chain_element_id": start_id,
        }
        return [
            {"event_type": EventType.SECTION_START, "time": start, "id": start_id, **d},
            *[e for l in children_events for e in l],
            # to do: Is this really needed by the pulse sheet viewer?
            {"event_type": EventType.LOOP_END, "time": start + loop_ir.length, **d},
            {"event_type": EventType.SECTION_END, "time": start + loop_ir.length, **d},
        ]

    def visit_MatchIR(
        self,
        match_ir: ir_mod.MatchIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert match_ir.length is not None
        events = self.visit_SectionIR(match_ir, start, max_events)
        if len(events) == 0:
            return []
        section_start_event = events[0]
        assert section_start_event["event_type"] == EventType.SECTION_START
        if match_ir.handle is not None:
            section_start_event["handle"] = match_ir.handle
            section_start_event["local"] = match_ir.local
        if match_ir.user_register is not None:
            section_start_event["user_register"] = match_ir.user_register
        if match_ir.prng_sample is not None:
            section_start_event["prng_sample"] = match_ir.prng_sample

        return events

    def visit_CaseIR(
        self,
        case_ir: ir_mod.CaseIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert case_ir.length is not None

        events = self.visit_SectionIR(case_ir, start, max_events)
        for e in events:
            e["state"] = case_ir.state
        return events

    def visit_LoopIterationIR(
        self,
        loop_iteration_ir: ir_mod.LoopIterationIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert loop_iteration_ir.length is not None
        common = {
            "section_name": loop_iteration_ir.section,
            "iteration": loop_iteration_ir.iteration,
            "num_repeats": loop_iteration_ir.num_repeats,
            "nesting_level": 0,
        }
        end = start + loop_iteration_ir.length

        max_events -= len(loop_iteration_ir.sweep_parameters)
        if loop_iteration_ir.iteration == 0:
            max_events -= 1

        # we'll add one LOOP_STEP_START, LOOP_STEP_END, LOOP_ITERATION_END each
        max_events -= 3

        children_events = self.generate_children_events(
            loop_iteration_ir, start, max_events
        )
        event_list = [
            {
                "event_type": EventType.LOOP_STEP_START,
                "time": start,
                **common,
            },
            *[e for l in children_events for e in l],
            {
                "event_type": EventType.LOOP_STEP_END,
                "time": end,
                **common,
            },
            *(
                [
                    {
                        "event_type": EventType.LOOP_ITERATION_END,
                        "time": end,
                        **common,
                    }
                ]
                if loop_iteration_ir.iteration == 0
                else []
            ),
        ]

        if loop_iteration_ir.shadow:
            for e in event_list:
                e["shadow"] = True
        return event_list

    def visit_LoopIterationPreambleIR(
        self,
        preamble: ir_mod.LoopIterationPreambleIR,
        start: int,
        max_events: int,
    ):
        # The preamble is not represented in the event list. We transparently pass through
        # all child events.

        events = []
        for child, child_start in zip(preamble.children, preamble.children_start):
            child_events = self.visit(child, start + child_start, max_events)
            max_events -= len(child_events)
            if max_events <= 0:
                break
            events.extend(child_events)

        return events

    def visit_PulseIR(
        self,
        pulse_ir: ir_mod.PulseIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert pulse_ir.length is not None
        params_list = [
            getattr(pulse_ir.pulse, f).uid
            for f in ("length", "amplitude", "phase")
            if isinstance(getattr(pulse_ir.pulse, f), ParameterInfo)
        ]
        if pulse_ir.pulse.pulse is not None:
            play_wave_id = pulse_ir.pulse.pulse.uid
        else:
            play_wave_id = "delay"

        start_id = next(self.id_tracker)
        d_start: dict[str, Any] = {
            "id": start_id,
        }
        d_end: dict[str, Any] = {"id": next(self.id_tracker)}
        d_common = {
            "section_name": pulse_ir.section,
            "signal": pulse_ir.pulse.signal.uid,
            "play_wave_id": play_wave_id,
            "chain_element_id": start_id,
            "parametrized_with": params_list,
        }

        if pulse_ir.amplitude is not None:
            d_start["amplitude"] = pulse_ir.amplitude

        if pulse_ir.phase is not None:
            d_start["phase"] = pulse_ir.phase

        if pulse_ir.amp_param_name:
            d_start["amplitude_parameter"] = pulse_ir.amp_param_name

        if pulse_ir.incr_phase_param_name:
            d_start["phase_increment_parameter"] = pulse_ir.incr_phase_param_name

        if pulse_ir.markers is not None and len(pulse_ir.markers) > 0:
            d_start["markers"] = [vars(m) for m in pulse_ir.markers]
        d_start["id_pulse_params"] = pulse_ir.pulse_pulse_params
        if pulse_ir.increment_oscillator_phase is not None:
            d_start["increment_oscillator_phase"] = pulse_ir.increment_oscillator_phase
        if pulse_ir.set_oscillator_phase is not None:
            d_start["set_oscillator_phase"] = pulse_ir.set_oscillator_phase

        is_delay = pulse_ir.pulse.pulse is None

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
                    "time": start + pulse_ir.length,
                    **d_end,
                    **d_common,
                },
            ]

        if pulse_ir.is_acquire:
            assert pulse_ir.integration_length is not None
            if pulse_ir.pulse.acquire_params is not None:
                d_start["acquire_handle"] = pulse_ir.pulse.acquire_params.handle
            else:
                d_start["acquire_handle"] = None
            return [
                {
                    "event_type": EventType.ACQUIRE_START,
                    "time": start,
                    **d_start,
                    **d_common,
                },
                {
                    "event_type": EventType.ACQUIRE_END,
                    "time": start + pulse_ir.integration_length,
                    **d_end,
                    **d_common,
                },
            ]

        return [
            {
                "event_type": EventType.PLAY_START,
                "time": start,
                **d_start,
                **d_common,
            },
            {
                "event_type": EventType.PLAY_END,
                "time": start + pulse_ir.length,
                **d_end,
                **d_common,
            },
        ]

    def visit_PhaseResetIR(
        self,
        ir: ir_mod.PhaseResetIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert ir.length is not None
        events = [
            {
                "event_type": EventType.RESET_HW_OSCILLATOR_PHASE,
                "time": start,
                "section_name": ir.section,
                "id": next(self.id_tracker),
                "device_id": device,
            }
            for device in ir.hw_osc_devices
        ]
        return events

    def generate_children_events(
        self,
        ir: ir_mod.IntervalIR,
        start: int,
        max_events: int | float,
        subsection_events=True,
    ) -> list[EventList]:
        assert ir.children_start is not None

        if not isinstance(ir, ir_mod.SectionIR):
            subsection_events = False

        if subsection_events:
            # take into account that we'll wrap with subsection events
            max_events -= 2 * len(ir.children)

        event_list_nested: list[EventList] = []
        assert ir.children_start is not None
        assert ir.length is not None
        for child_start, child in ir.iter_children():
            if max_events <= 0:
                break

            event_list_nested.append(self.visit(child, start + child_start, max_events))
            max_events -= len(event_list_nested[-1])

        # if event_list_nested was cut because max_events was exceeded, pad with empty
        # lists. This is necessary because the PSV requires the subsection events to be
        # present.
        # todo: investigate if this is a bug in the PSV.
        event_list_nested.extend(
            [] for _ in range(len(ir.children) - len(event_list_nested))
        )

        if subsection_events and isinstance(ir, ir_mod.SectionIR):
            # Wrap child sections in SUBSECTION_START & SUBSECTION_END.
            for i, child in enumerate(ir.children):
                if isinstance(child, ir_mod.SectionIR):
                    assert child.length is not None
                    start_id = next(self.id_tracker)
                    d = {"section_name": ir.section, "chain_element_id": start_id}
                    event_list_nested[i] = [
                        {
                            "event_type": EventType.SUBSECTION_START,
                            "time": ir.children_start[i] + start,
                            "subsection_name": child.section,
                            "id": start_id,
                            **d,
                        },
                        *event_list_nested[i],
                        {
                            "event_type": EventType.SUBSECTION_END,
                            "time": ir.children_start[i] + child.length + start,
                            "subsection_name": child.section,
                            "id": next(self.id_tracker),
                            **d,
                        },
                    ]

        return event_list_nested


def generate_event_list_from_ir(
    ir: ir_mod.IRTree,
    expand_loops: bool,
    max_events: int | float,
) -> EventList:
    id_tracker = itertools.count()
    event_list = EventListGenerator(
        id_tracker=id_tracker,
        expand_loops=expand_loops,
    ).run(ir.root, start=0, max_events=max_events)
    for event in event_list:
        if "id" not in event:
            event["id"] = next(id_tracker)
        event["time"] = event["time"] * TINYSAMPLE
    return event_list
