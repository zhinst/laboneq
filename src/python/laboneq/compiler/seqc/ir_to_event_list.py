# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import itertools
from typing import Iterator, TYPE_CHECKING

from laboneq.compiler.common.compiler_settings import CompilerSettings, TINYSAMPLE
from laboneq.compiler.common import awg_info
from laboneq.compiler.event_list.event_type import EventList, EventType
from laboneq.compiler.event_list import event_list_generator as event_gen
from laboneq.compiler import ir as ir_def
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.seqc import ir as ir_seqc
from laboneq.compiler.common.pulse_parameters import encode_pulse_parameters
from laboneq.compiler.common.play_wave_type import PlayWaveType

if TYPE_CHECKING:
    from laboneq.compiler.seqc.passes.oscillator_parameters import (
        SoftwareOscillatorParameters,
    )


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
        for event in event_list:
            if "id" not in event:
                event["id"] = next(id_tracker)
    for event in event_list:
        event["time"] = event["time"] * TINYSAMPLE
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

        if (
            pulse_ir.pulse.signal.oscillator
            and pulse_ir.pulse.signal.oscillator.is_hardware
        ):
            if pulse_ir.increment_oscillator_phase is not None:
                d_start["increment_oscillator_phase"] = (
                    pulse_ir.increment_oscillator_phase
                )
            if pulse_ir.set_oscillator_phase is not None:
                if (
                    pulse_ir.pulse.signal.oscillator
                    and pulse_ir.pulse.signal.oscillator.is_hardware
                ):
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


def _apply_pulse_parameters(
    event_list: EventList, osc_times: SoftwareOscillatorParameters
):
    """Apply pulse parameters to the event list in-place."""
    # TODO: Remove once path to pulse generation from IR is ok.
    for event in event_list:
        if event["event_type"] in [EventType.PLAY_START, EventType.ACQUIRE_START]:
            if event["signal"] not in osc_times.freq_keys():
                continue
            event["oscillator_frequency"] = osc_times.freq_at(
                event["signal"], event["time"]
            )
        if event["event_type"] in [EventType.PLAY_START, EventType.DELAY_START]:
            val = osc_times.phase_at(event["signal"], event["time"])
            if val is not None:
                if "increment_oscillator_phase" in event:
                    del event["increment_oscillator_phase"]
            event["oscillator_phase"] = val


def event_list_per_awg(
    tree: ir_def.IRTree,
    settings: CompilerSettings,
    osc_params: SoftwareOscillatorParameters,
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
        _apply_pulse_parameters(event_list, osc_params)
        for event in event_list:
            if "id" not in event:
                event["id"] = next(id_tracker)
            event["time"] = event["time"] * TINYSAMPLE
        event_lists_by_awg[awg_ir.awg.key] = event_list
    return event_lists_by_awg
