# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import itertools
import math
from typing import Iterator, TYPE_CHECKING

from laboneq.compiler.common.compiler_settings import CompilerSettings, TINYSAMPLE
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common import awg_info
from laboneq.compiler.event_list.event_type import EventList, EventType
from laboneq.compiler.event_list import event_list_generator as event_gen
from laboneq.compiler import ir as ir_def
from laboneq.compiler.seqc import ir as ir_seqc
from laboneq.compiler.common.pulse_parameters import encode_pulse_parameters
from laboneq.compiler.common.play_wave_type import PlayWaveType

if TYPE_CHECKING:
    from laboneq.compiler.seqc.passes.oscillator_parameters import OscillatorParameters


def _apply_frequencies(event_list: EventList, osc_times: OscillatorParameters):
    """Apply software oscillator frequencies to the event list in-place."""
    # TODO: Remove once path to pulse generation from IR is ok.
    events = [
        e
        for e in event_list
        if e["event_type"] in [EventType.PLAY_START, EventType.ACQUIRE_START]
    ]
    for event in events:
        if event["signal"] not in osc_times.freq_keys():
            continue
        event["oscillator_frequency"] = osc_times.freq_at(
            event["signal"], event["time"]
        )


def _calculate_osc_phase(event_list: EventList, ir: ir_def.IRTree):
    """Traverse the event list, and elaborate the phase of each played pulse.

    For SW oscillators, calculate the time since the last set/reset of that oscillator,
    and store it in the event as `oscillator_phase`. Illegal phase sets/resets in
    conditional branches have previously been ruled out (see scheduler).
    The `[increment|set]_oscillator_phase` fields are removed if present, and their
    effect is aggregated into `oscillator_phase`.

    For HW oscillators, do nothing. Absolute phase sets are illegal (and were caught in
    the scheduler), and phase increments will be handled in the code generator.

    After this function returns, all play events will contain the following phase-related
    fields:
     - "phase": the baseband phase of the pulse
     - "oscillator_phase": the oscillator phase for SW modulators, `None` for HW
     - "increment_oscillator_phase": if present, the event should increment the HW modulator
    """
    # TODO: Remove once path to pulse generation from IR is ok
    oscillator_phase_cumulative = {}
    oscillator_phase_sets = {}
    phase_reset_time = 0.0
    priority_map = {
        EventType.PLAY_START: 0,
        EventType.DELAY_START: 0,
        EventType.RESET_SW_OSCILLATOR_PHASE: -15,
    }
    sorted_events = sorted(
        (e for e in event_list if e["event_type"] in priority_map),
        key=lambda e: (e["time"], priority_map[e["event_type"]]),
    )

    oscillator_map = {signal.uid: signal.oscillator for signal in ir.signals}
    device_map = {signal.uid: signal.device for signal in ir.signals}

    for event in sorted_events:
        if event["event_type"] == EventType.RESET_SW_OSCILLATOR_PHASE:
            phase_reset_time = event["time"]
            for signal_id in oscillator_phase_cumulative.keys():
                oscillator_phase_cumulative[signal_id] = 0.0

        else:
            signal_id = event["signal"]
            oscillator_info = oscillator_map[signal_id]
            is_hw_osc = oscillator_info.is_hardware if oscillator_info else False
            if (phase_incr := event.get("increment_oscillator_phase")) is not None:
                if not is_hw_osc:
                    if signal_id not in oscillator_phase_cumulative:
                        oscillator_phase_cumulative[signal_id] = 0.0
                    oscillator_phase_cumulative[signal_id] += phase_incr
                    del event["increment_oscillator_phase"]

            # if both "increment_oscillator_phase" and "set_oscillator_phase" are specified,
            # the absolute phase overwrites the increment.
            if (osc_phase := event.get("set_oscillator_phase")) is not None:
                assert not oscillator_info.is_hardware, (
                    "cannot set phase of HW oscillators (should have been caught earlier)"
                )
                oscillator_phase_cumulative[signal_id] = osc_phase
                oscillator_phase_sets[signal_id] = event["time"]
                del event["set_oscillator_phase"]

            if is_hw_osc:
                event["oscillator_phase"] = None
            else:  # SW oscillator
                device = device_map[signal_id]
                device_type = DeviceType.from_device_info_type(device.device_type)
                if not device_type.is_qa_device:
                    incremented_phase = oscillator_phase_cumulative.get(signal_id, 0.0)
                    phase_reference_time = max(
                        phase_reset_time, oscillator_phase_sets.get(signal_id, 0.0)
                    )
                    oscillator_frequency = event.get("oscillator_frequency", 0.0)
                    t = event["time"] - phase_reference_time
                    event["oscillator_phase"] = (
                        t * 2.0 * math.pi * oscillator_frequency + incremented_phase
                    )
                else:
                    event["oscillator_phase"] = 0.0


def _create_start_events(devices: list[ir_def.DeviceIR]) -> EventList:
    retval = []

    # Add initial events to reset the NCOs.
    # Todo (PW): Drop once system tests have been migrated from legacy behaviour.
    for device_info in devices:
        try:
            device_type = DeviceType.from_device_info_type(  # @IgnoreException
                device_info.device_type
            )
        except ValueError:
            # Not every device has a corresponding DeviceType (e.g. PQSC)
            continue
        if not device_type.supports_reset_osc_phase:
            continue
        retval.append(
            {
                "event_type": EventType.INITIAL_RESET_HW_OSCILLATOR_PHASE,
                "device_id": device_info.uid,
                "duration": device_type.reset_osc_duration,
                "time": 0,
            }
        )
    return retval


def generate_event_list_from_ir(
    ir: ir_def.IRTree,
    settings: CompilerSettings,
    expand_loops: bool,
    max_events: int,
) -> EventList:
    event_list = _create_start_events(ir.devices)
    if ir.root is not None:
        id_tracker = itertools.count()
        event_list.extend(
            event_gen.EventListGenerator().run(
                ir.root,
                start=0,
                max_events=max_events,
                id_tracker=id_tracker,
                expand_loops=expand_loops,
                settings=settings,
            )
        )
        # assign every event an id
        for event in event_list:
            if "id" not in event:
                event["id"] = next(id_tracker)
    # convert time from units of tiny samples to seconds
    for event in event_list:
        event["time"] = event["time"] * TINYSAMPLE
    # TODO: Move to oscillator params pass
    _calculate_osc_phase(event_list, ir)
    return event_list


class EventListGeneratorCodeGenerator(event_gen.EventListGenerator):
    def visit_SingleAwgIR(
        self,
        ir: ir_seqc.SingleAwgIR,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> EventList:
        return [
            e
            for l in self.generate_children_events(
                ir, start, max_events, settings, id_tracker, expand_loops
            )
            for e in l
        ]

    def visit_PulseIR(
        self,
        pulse_ir: ir_def.PulseIR,
        start: int,
        _max_events: int,
        id_tracker: Iterator[int],
        _expand_loops,
        _settings: CompilerSettings,
    ) -> EventList:
        assert pulse_ir.length is not None
        if pulse_ir.pulse.pulse is not None:
            play_wave_id = pulse_ir.pulse.pulse.uid
        else:
            play_wave_id = "delay"

        start_id = next(id_tracker)
        d_start = {"id": start_id}
        d_end = {"id": next(id_tracker)}
        d_common = {
            "section_name": pulse_ir.section,
            "signal": pulse_ir.pulse.signal.uid,
            "play_wave_id": play_wave_id,
            "chain_element_id": start_id,
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

        if pulse_ir.pulse_pulse_params:
            d_start["pulse_pulse_parameters"] = encode_pulse_parameters(
                pulse_ir.pulse_pulse_params
            )
        if pulse_ir.play_pulse_params:
            d_start["play_pulse_parameters"] = encode_pulse_parameters(
                pulse_ir.play_pulse_params
            )

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
            if pulse_ir.pulse.acquire_params is not None:
                d_start["acquisition_type"] = [
                    pulse_ir.pulse.acquire_params.acquisition_type
                ]
                d_start["acquire_handle"] = pulse_ir.pulse.acquire_params.handle
            else:
                d_start["acquisition_type"] = []
                d_start["acquire_handle"] = None
            return [
                {
                    "event_type": EventType.ACQUIRE_START,
                    "time": start + pulse_ir.offset,
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
                "time": start + pulse_ir.offset,
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

    def visit_AcquireGroupIR(
        self,
        acquire_group_ir: ir_def.AcquireGroupIR,
        start: int,
        _max_events: int,
        id_tracker: Iterator[int],
        _expand_loops,
        _settings: CompilerSettings,
    ) -> EventList:
        assert acquire_group_ir.length is not None
        assert (
            len(acquire_group_ir.pulses)
            == len(acquire_group_ir.amplitudes)
            == len(acquire_group_ir.play_pulse_params)
            == len(acquire_group_ir.pulse_pulse_params)
        )

        assert all(
            acquire_group_ir.pulses[0].acquire_params.handle == p.acquire_params.handle
            for p in acquire_group_ir.pulses
        )
        assert all(
            acquire_group_ir.pulses[0].acquire_params.acquisition_type
            == p.acquire_params.acquisition_type
            for p in acquire_group_ir.pulses
        )
        start_id = next(id_tracker)
        signal_id = acquire_group_ir.pulses[0].signal.uid
        assert all(p.signal.uid == signal_id for p in acquire_group_ir.pulses)
        d = {
            "section_name": acquire_group_ir.section,
            "signal": signal_id,
            "play_wave_id": [p.pulse.uid for p in acquire_group_ir.pulses],
            "amplitude": acquire_group_ir.amplitudes,
            "chain_element_id": start_id,
            "acquisition_type": [
                acquire_group_ir.pulses[0].acquire_params.acquisition_type
            ],
            "acquire_handle": acquire_group_ir.pulses[0].acquire_params.handle,
        }

        if acquire_group_ir.pulse_pulse_params:
            d["pulse_pulse_parameters"] = [
                encode_pulse_parameters(par) if par is not None else None
                for par in acquire_group_ir.pulse_pulse_params
            ]
        if acquire_group_ir.play_pulse_params:
            d["play_pulse_parameters"] = [
                encode_pulse_parameters(par) if par is not None else None
                for par in acquire_group_ir.play_pulse_params
            ]
        return [
            {
                "event_type": EventType.ACQUIRE_START,
                "time": start + acquire_group_ir.offset,
                "id": start_id,
                **d,
            },
            {
                "event_type": EventType.ACQUIRE_END,
                "time": start + acquire_group_ir.length,
                "id": next(id_tracker),
                **d,
            },
        ]

    def visit_CaseIR(
        self,
        case_ir: ir_def.CaseIR,
        start: int,
        max_events: int,
        id_tracker: Iterator[int],
        expand_loops,
        settings: CompilerSettings,
    ) -> EventList:
        if not case_ir.children:
            return []
        return super().visit_CaseIR(
            case_ir, start, max_events, id_tracker, expand_loops, settings
        )


def event_list_per_awg(
    tree: ir_def.IRTree, settings: CompilerSettings, osc_params: OscillatorParameters
) -> dict[awg_info.AwgKey, EventList]:
    """Generate event list per AWG in the tree root."""
    event_lists_by_awg = {}
    id_tracker = itertools.count()
    for awg_ir in tree.root.children:
        assert isinstance(awg_ir, ir_seqc.SingleAwgIR)
        event_list = _create_start_events(
            [dev for dev in tree.devices if dev.uid == awg_ir.awg.device_id]
        )
        event_list.extend(
            EventListGeneratorCodeGenerator().run(
                awg_ir,
                start=0,
                settings=settings,
                max_events=float("inf"),
                expand_loops=False,
                id_tracker=id_tracker,
            )
        )
        _apply_frequencies(event_list, osc_params)
        for event in event_list:
            if "id" not in event:
                # assign every event an id
                event["id"] = next(id_tracker)
            # convert time from units of tiny samples to seconds
            event["time"] = event["time"] * TINYSAMPLE
        _calculate_osc_phase(event_list, tree)
        event_lists_by_awg[awg_ir.awg.key] = event_list
    return event_lists_by_awg
