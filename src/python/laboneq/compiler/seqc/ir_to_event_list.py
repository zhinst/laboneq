# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

from laboneq.compiler import ir as ir_def
from laboneq.compiler.common import awg_info
from laboneq.compiler.common.compiler_settings import TINYSAMPLE, CompilerSettings
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.event_list import event_list_generator as event_gen
from laboneq.compiler.event_list.event_type import EventList, EventType
from laboneq.compiler.seqc import ir as ir_seqc

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
            assert device_info.device_type is not None
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
) -> tuple[EventList, event_gen.IrPositionMap]:
    event_list = _create_start_events(ir.devices)
    ir_id_map = event_gen.IrPositionMapper().create_ir_position_map(ir.root)
    if ir.root is not None:
        id_tracker = itertools.count()
        events = event_gen.EventListGenerator(
            id_tracker=id_tracker,
            expand_loops=expand_loops,
            settings=settings,
            ir_id_map=ir_id_map,
        ).run(ir.root, start=0, max_events=max_events)
        event_list.extend(events)
        for event in event_list:
            if "id" not in event:
                event["id"] = next(id_tracker)
    for event in event_list:
        event["time"] = event["time"] * TINYSAMPLE
    return event_list, ir_id_map


class EventListGeneratorCodeGenerator(event_gen.EventListGenerator):
    def visit_SingleAwgIR(
        self,
        ir: ir_seqc.SingleAwgIR,
        start: int,
        max_events: int,
    ) -> EventList:
        return [
            e for l in self.generate_children_events(ir, start, max_events) for e in l
        ]

    def visit_PulseIR(
        self,
        pulse_ir: ir_def.PulseIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert pulse_ir.length is not None
        if not pulse_ir.is_acquire or pulse_ir.pulse.pulse is None:
            return []
        play_wave_id = pulse_ir.pulse.pulse.uid
        start_id = next(self.id_tracker)
        d_start: dict[str, Any] = {
            "id": start_id,
            "position": self.ir_id_map[id(pulse_ir)].start,
        }
        d_end: dict[str, Any] = {"id": next(self.id_tracker)}
        d_common = {
            "section_name": pulse_ir.section,
            "signal": pulse_ir.pulse.signal.uid,
            "play_wave_id": play_wave_id,
            "chain_element_id": start_id,
        }
        if pulse_ir.markers is not None and len(pulse_ir.markers) > 0:
            d_start["markers"] = [vars(m) for m in pulse_ir.markers]
        d_start["id_pulse_params"] = pulse_ir.pulse_pulse_params
        assert pulse_ir.integration_length is not None
        if pulse_ir.pulse.acquire_params is not None:
            d_start["acquire_handle"] = pulse_ir.pulse.acquire_params.handle
        else:
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

    def visit_AcquireGroupIR(
        self,
        acquire_group_ir: ir_def.AcquireGroupIR,
        start: int,
        max_events: int,
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
        start_id = next(self.id_tracker)
        signal_id = acquire_group_ir.pulses[0].signal.uid
        assert all(p.signal.uid == signal_id for p in acquire_group_ir.pulses)
        assert all(p.pulse is not None for p in acquire_group_ir.pulses)
        d = {
            "section_name": acquire_group_ir.section,
            "signal": signal_id,
            "play_wave_id": [p.pulse.uid for p in acquire_group_ir.pulses],
            "chain_element_id": start_id,
            "acquire_handle": acquire_group_ir.pulses[0].acquire_params.handle,
        }
        d["id_pulse_params"] = acquire_group_ir.pulse_pulse_params
        return [
            {
                "event_type": EventType.ACQUIRE_START,
                "time": start + acquire_group_ir.offset,
                "id": start_id,
                "position": self.ir_id_map[id(acquire_group_ir)].start,
                **d,
            },
            {
                "event_type": EventType.ACQUIRE_END,
                "time": start + acquire_group_ir.length,
                "id": next(self.id_tracker),
                **d,
            },
        ]

    def visit_MatchIR(
        self,
        match_ir: ir_def.MatchIR,
        start: int,
        max_events: int,
    ) -> EventList:
        assert match_ir.length is not None
        return self.visit_SectionIR(match_ir, start, max_events)

    def visit_CaseIR(
        self,
        case_ir: ir_def.CaseIR,
        start: int,
        max_events: int,
    ) -> EventList:
        return self.visit_SectionIR(case_ir, start, max_events)


def _apply_pulse_parameters(
    event_list: EventList, osc_times: SoftwareOscillatorParameters
):
    """Apply pulse parameters to the event list in-place."""
    # TODO: Remove once path to pulse generation from IR is ok.
    for event in event_list:
        if event["event_type"] == EventType.ACQUIRE_START:
            if event["signal"] not in osc_times.freq_keys():
                continue
            event["oscillator_frequency"] = osc_times.freq_at(
                event["signal"], event["time"]
            )


def event_list_per_awg(
    tree: ir_def.IRTree,
    settings: CompilerSettings,
    osc_params: SoftwareOscillatorParameters,
) -> tuple[
    dict[awg_info.AwgKey, EventList], dict[awg_info.AwgKey, event_gen.IrPositionMapper]
]:
    """Generate event list per AWG in the tree root."""
    assert tree.root is not None
    event_lists_by_awg: dict[awg_info.AwgKey, EventList] = {}
    ir_id_map_by_awg: dict[awg_info.AwgKey, event_gen.IrPositionMapper] = {}
    id_tracker = itertools.count()

    for awg_ir in tree.root.children:
        assert isinstance(awg_ir, ir_seqc.SingleAwgIR)
        ir_id_map = event_gen.IrPositionMapper().create_ir_position_map(awg_ir)
        event_list_generator = EventListGeneratorCodeGenerator(
            expand_loops=False,
            settings=settings,
            id_tracker=id_tracker,
            ir_id_map=ir_id_map,
        )
        event_list = _create_start_events(
            [dev for dev in tree.devices if dev.uid == awg_ir.awg.device_id]
        )
        event_list.extend(event_list_generator.run(awg_ir, start=0))
        _apply_pulse_parameters(event_list, osc_params)
        for event in event_list:
            if "id" not in event:
                event["id"] = next(id_tracker)
            event["time"] = event["time"] * TINYSAMPLE
        event_lists_by_awg[awg_ir.awg.key] = event_list
        ir_id_map_by_awg[awg_ir.awg.key] = ir_id_map
    return event_lists_by_awg, ir_id_map_by_awg
