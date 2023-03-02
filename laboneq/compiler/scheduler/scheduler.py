# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from box import Box
from engineering_notation import EngNumber
from sortedcollections import ValueSortedDict
from sortedcontainers import SortedDict

from laboneq._observability.tracing import trace
from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.device_type import DeviceType
from laboneq.compiler.common.event_type import EventType
from laboneq.compiler.common.play_wave_type import PlayWaveType
from laboneq.compiler.experiment_access.experiment_dao import (
    ExperimentDAO,
    ParamRef,
    PulseDef,
)
from laboneq.compiler.experiment_access.section_graph import SectionGraph
from laboneq.compiler.scheduler.event_graph import EventGraph, EventRelation, node_info
from laboneq.compiler.scheduler.event_graph_builder import (
    ChainElement,
    EventGraphBuilder,
)
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.core.exceptions import LabOneQException

_logger = logging.getLogger(__name__)


@dataclass
class _PlayWave:
    id: str
    signal: str
    length: float = None
    offset: Any = None
    amplitude: Any = None
    parameterized_with: list = field(default_factory=list)
    acquire_handle: str = None
    acquisition_type: list = field(default_factory=list)
    phase: float = None
    increment_oscillator_phase: float = None
    set_oscillator_phase: float = None
    oscillator_frequency: float = None
    play_wave_type: PlayWaveType = PlayWaveType.PLAY
    play_pulse_parameters: Dict[str, Any] = field(default_factory=dict)
    pulse_pulse_parameters: Dict[str, Any] = field(default_factory=dict)
    precompensation_clear: bool = field(default=False)
    markers: Any = None


@dataclass
class _PreambleInserter:
    event_graph: EventGraph
    before_loops_id: int
    loop_step_start_id: int
    previous_loop_step_end_id: Optional[int]

    def insert(
        self,
        node_ids: Union[int, List[int]],
        add_before_first_loop_start: bool,
    ):
        if not isinstance(node_ids, list):
            node_ids = [node_ids]
        for n in node_ids:
            if self.previous_loop_step_end_id is None:
                # First loop step: Schedule before/after start of outer looping section
                if add_before_first_loop_start:
                    self.event_graph.after_or_at(self.before_loops_id, n)
                else:
                    self.event_graph.after_or_at(n, self.before_loops_id)
            else:
                # After first loop: Schedule after beginning of new loop step
                self.event_graph.after_or_at(n, self.previous_loop_step_end_id)
            self.event_graph.after_or_at(self.loop_step_start_id, n)


def _set_playwave_resolved_param(section_pulse, play_wave: _PlayWave, name, value=None):
    if getattr(section_pulse, name) is not None:
        setattr(play_wave, name, getattr(section_pulse, name))
    elif value is not None:
        setattr(play_wave, name, value)
    if getattr(section_pulse, name + "_param") is not None:
        param_ref = ParamRef(getattr(section_pulse, name + "_param"))
        setattr(play_wave, name, param_ref)
        play_wave.parameterized_with.append(param_ref)


def _assign_playwave_parameters(play_wave, section_pulse, oscillator, length):
    play_wave.parameterized_with = []
    _set_playwave_resolved_param(section_pulse, play_wave, "offset")
    _set_playwave_resolved_param(section_pulse, play_wave, "amplitude")
    _set_playwave_resolved_param(section_pulse, play_wave, "length", length)
    _set_playwave_resolved_param(section_pulse, play_wave, "phase")
    _set_playwave_resolved_param(section_pulse, play_wave, "increment_oscillator_phase")
    _set_playwave_resolved_param(section_pulse, play_wave, "set_oscillator_phase")
    if oscillator is None or oscillator.hardware:
        pass
    elif oscillator.frequency is not None:
        play_wave.oscillator_frequency = oscillator.frequency
    elif oscillator.frequency_param is not None:
        param_ref = ParamRef(oscillator.frequency_param)
        play_wave.oscillator_frequency = param_ref
        play_wave.parameterized_with.append(param_ref)

    play_wave.precompensation_clear = section_pulse.precompensation_clear
    play_pulse_parameters = section_pulse.play_pulse_parameters
    pulse_pulse_parameters = section_pulse.pulse_pulse_parameters
    if play_pulse_parameters is not None:
        for param, val in play_pulse_parameters.items():
            if isinstance(val, (float, int, complex)):
                play_wave.play_pulse_parameters[param] = val
            else:
                param_ref = ParamRef(val)
                play_wave.play_pulse_parameters[param] = param_ref
                play_wave.parameterized_with.append(param_ref)
    if pulse_pulse_parameters is not None:
        for param, val in pulse_pulse_parameters.items():
            if isinstance(val, (float, int, complex)):
                play_wave.pulse_pulse_parameters[param] = val
            else:
                param_ref = ParamRef(val)
                play_wave.pulse_pulse_parameters[param] = param_ref
                play_wave.parameterized_with.append(param_ref)

    if section_pulse.acquire_params is not None:
        play_wave.acquire_handle = section_pulse.acquire_params.handle
        play_wave.acquisition_type.append(section_pulse.acquire_params.acquisition_type)


class Scheduler:
    def __init__(
        self,
        experiment_dao: ExperimentDAO,
        section_graph: SectionGraph,
        sampling_rate_tracker: SamplingRateTracker,
        clock_settings: Optional[Dict] = None,
        settings: Optional[CompilerSettings] = None,
    ):
        self._experiment_dao = experiment_dao
        self._section_graph_object = section_graph
        self._sampling_rate_tracker = sampling_rate_tracker
        self._clock_settings = clock_settings or {}
        self._settings = settings or CompilerSettings()
        self._event_graph = EventGraph()
        self._section_grids = {}
        self._section_events = {}

        self._loop_step_events: Dict[str, Dict[int, Tuple[int, int, int]]] = {}

    @trace("scheduler.run()", {"version": "v1"})
    def run(self):
        root_sections = self._section_graph_object.root_sections()
        start_events = self._add_start_events()

        EventGraphBuilder.build_section_structure(
            self._event_graph, self._section_graph_object, start_events
        )

        match_case_signals = {}
        empty_match_case_sections = {}
        clear_bit_events_of_section = {}
        for section_node in self._section_graph_object.topologically_sorted_sections():

            parent = self._section_graph_object.parent(section_node)
            assert parent is not None or section_node in root_sections
            section_info = self._section_graph_object.section_info(section_node)
            section_name = section_info.section_id

            section_span = self._event_graph.find_section_start_end(section_name)

            section_start_node = section_span.start

            signals = self._experiment_dao.section_signals(
                section_info.section_display_name
            )

            clear_bit_events = []
            parent_section_trigger_states = set()
            parent_section_id = section_name
            while True:
                parent_section_id = self._experiment_dao.section_parent(
                    parent_section_id
                )
                if parent_section_id is None:
                    break
                parent_section_info = self._experiment_dao.section_info(
                    parent_section_id
                )
                for trigger_info in parent_section_info.trigger_output:
                    state = int(trigger_info["state"])
                    if state:
                        parent_section_trigger_states.add(state)

            for trigger_info in section_info.trigger_output:
                state = int(trigger_info["state"])
                if state:
                    for bit in range(2):
                        bit_mask = 2**bit
                        if bit_mask & state > 0:
                            # check if the current bit is controlled by any parent section
                            # if so, leave it alone
                            if not any(
                                [
                                    parent_statee & bit_mask > 0
                                    for parent_statee in parent_section_trigger_states
                                ]
                            ):
                                set_bit_event = self._event_graph.add_node(
                                    event_type=EventType.DIGITAL_SIGNAL_STATE_CHANGE,
                                    section_name=section_node,
                                    bit=bit,
                                    change="SET",
                                    signal=trigger_info["signal_id"],
                                )
                                self._event_graph.after_or_at(
                                    set_bit_event, section_span.start
                                )
                                self._event_graph.after_or_at(
                                    section_span.end, set_bit_event
                                )
                                clear_bit_event = self._event_graph.add_node(
                                    event_type=EventType.DIGITAL_SIGNAL_STATE_CHANGE,
                                    section_name=section_node,
                                    bit=bit,
                                    change="CLEAR",
                                    signal=trigger_info["signal_id"],
                                )
                                self._event_graph.after_or_at(
                                    clear_bit_event, section_span.start
                                )
                                self._event_graph.after_or_at(
                                    section_span.end, clear_bit_event
                                )
                                clear_bit_events.append(clear_bit_event)

            if section_info.state is not None:
                if signals:
                    match_case_signals.setdefault(parent, set()).update(signals)
                else:
                    empty_match_case_sections[section_name] = (
                        parent,
                        section_start_node,
                    )

            for signal_id in signals:
                _logger.debug(
                    "Considering signal %s in section %s", signal_id, section_name
                )

                signal_info_main = self._experiment_dao.signal_info(signal_id)

                _logger.debug("signal_info_main =  %s", signal_info_main)
                play_wave_chain: List[_PlayWave] = []

                is_integration = signal_info_main.signal_type == "integration"

                device_id = signal_info_main.device_id

                sampling_rate = self._sampling_rate_tracker.sampling_rate_for_device(
                    device_id
                )

                if section_info.state is not None:
                    # Add an additional anchor to time the executeTableEntry evaluation:
                    play_wave_chain.append(
                        _PlayWave(
                            id="CASE_EVALUATION",
                            signal=signal_id,
                            play_wave_type=PlayWaveType.CASE_EVALUATION,
                            offset=32 / sampling_rate,
                        )
                    )
                for pulse_index, section_pulse in enumerate(
                    self._experiment_dao.section_pulses(
                        section_info.section_display_name, signal_id
                    )
                ):
                    pulse_name = section_pulse.pulse_id
                    pulse_def = None
                    if pulse_name is not None:
                        pulse_def = self._experiment_dao.pulse(pulse_name)
                        play_wave = _PlayWave(
                            id=pulse_name,
                            signal=signal_id,
                            play_wave_type=PlayWaveType.INTEGRATION
                            if is_integration
                            else PlayWaveType.PLAY,
                            markers=section_pulse.markers,
                        )
                    else:
                        play_wave = _PlayWave(
                            id="DELAY",
                            signal=signal_id,
                            play_wave_type=PlayWaveType.DELAY,
                        )

                    assert signal_id == signal_info_main.signal_id

                    length = PulseDef.effective_length(pulse_def, sampling_rate)
                    oscillator = self._experiment_dao.signal_oscillator(signal_id)
                    _assign_playwave_parameters(
                        play_wave, section_pulse, oscillator, length
                    )

                    play_wave_chain.append(play_wave)
                self.add_play_wave_chain(
                    play_wave_chain,
                    section_info.section_id,
                    right_aligned=(section_info.align == "right"),
                    section_start_node=section_start_node,
                )

            clear_bit_events_of_section[section_name] = clear_bit_events

        for section_name, clear_bit_events in clear_bit_events_of_section.items():
            section_span = self._event_graph.find_section_start_end(section_name)
            for edge in self._event_graph.out_edges(section_span.end):
                # make sure that the clearing of the trigger event happens after everything else in the section
                # while also making sure there is a path in the graph from section end -> bit clear node -> section start
                # so that the event gets properly copied in loops
                for clear_bit_event in clear_bit_events:
                    other_node = edge[1]
                    if other_node != clear_bit_event:
                        if edge[2]["relation"] == EventRelation.AFTER_OR_AT:
                            self._event_graph.after_or_at(clear_bit_event, other_node)

        for empty_section, parent in empty_match_case_sections.items():
            for signal in match_case_signals[parent[0]]:
                # This is an empty branch of a feedback, but we still want to create
                # events such that we can play zeros during the duration of that event
                # via the command table - it needs an entry even when nothing is played.

                signal_info_main = self._experiment_dao.signal_info(signal)
                sampling_rate = self._sampling_rate_tracker.sampling_rate_for_device(
                    signal_info_main.device_id
                )
                timing_anchor = _PlayWave(
                    id="CASE_EVALUATION",
                    signal=signal,
                    play_wave_type=PlayWaveType.CASE_EVALUATION,
                    offset=32 / sampling_rate,
                )
                play_zeros = _PlayWave(
                    id="EMPTY_MATCH_CASE_DELAY",
                    play_wave_type=PlayWaveType.EMPTY_CASE,
                    signal=signal,
                    length=1e-9,
                )
                self.add_play_wave_chain(
                    [timing_anchor, play_zeros],
                    parent_section_name=empty_section,
                    right_aligned=False,
                    section_start_node=parent[1],
                )

        repeat_sections: Dict[str, Scheduler.RepeatSectionsEntry] = {}
        for section_node in list(
            reversed(list(self._section_graph_object.topologically_sorted_sections()))
        ):
            _logger.debug("Processing section %s", section_node)
            section_info_1 = self._section_graph_object.section_info(section_node)

            if section_info_1.has_repeat:
                section_name = section_info_1.section_id
                _logger.debug("Adding repeat for section %s", section_name)
                parameters_list = [
                    {
                        "id": param["id"],
                        "start": param["start"],
                        "step": param["step"],
                        "values": param["values"],
                    }
                    for param in self._experiment_dao.section_parameters(
                        section_info_1.section_display_name
                    )
                ]
                num_repeats = section_info_1.count

                reset_phase_hw = section_info_1.reset_oscillator_phase
                reset_phase_sw = (
                    reset_phase_hw or section_info_1.averaging_type == "hardware"
                )
                repeat_sections[section_name] = self.add_repeat(
                    section_name,
                    num_repeats,
                    parameters_list,
                    reset_phase_sw,
                    reset_phase_hw,
                )

        EventGraphBuilder.complete_section_structure(
            self._event_graph, self._section_graph_object
        )

        self.generate_loop_events(repeat_sections)
        self._add_repetition_time_edges()
        self._sorted_events = self._event_graph.sorted_events()
        self._add_feedback_time_edges()
        self.calculate_timing()
        self.verify_timing()
        _logger.debug("Calculating play wave parameters")
        self._calculate_play_wave_parameters()

    def _calculate_play_wave_parameters(self):
        amplitude_resolution = pow(2, self._settings.AMPLITUDE_RESOLUTION_BITS)
        parameter_values = {}
        oscillator_phase_cumulative = {}
        oscillator_phase_sets = {}

        phase_reset_time = 0.0
        sorted_events = SortedDict()
        priority_map = {
            EventType.SET_OSCILLATOR_PHASE: -5,
            EventType.PLAY_START: 0,
            EventType.ACQUIRE_START: 0,
            EventType.PARAMETER_SET: -15,
            EventType.INCREMENT_OSCILLATOR_PHASE: -9,
            EventType.RESET_SW_OSCILLATOR_PHASE: -15,
        }

        for local_event_id in self._event_timing.keys():
            event = self._event_graph.node(local_event_id)

            if event["event_type"] in priority_map:
                key = (event["time"], priority_map[event["event_type"]], event["id"])
                sorted_events[key] = event

        def make_error_nt_param(param: ParamRef, event):
            return LabOneQException(
                f"Parameter {param.param_name} in section {event['section_name']} "
                f"could not be resolved. "
                f"Note that only RT sweep parameters are currently supported here."
            )

        for event in sorted_events.values():
            if event["event_type"] == EventType.PARAMETER_SET:
                param_obj = event["parameter"]
                parameter_values[param_obj["id"]] = self._event_graph.node(event["id"])[
                    "value"
                ]
            if event["event_type"] == EventType.RESET_SW_OSCILLATOR_PHASE:
                phase_reset_time = event["time"]

            if event["event_type"] == EventType.INCREMENT_OSCILLATOR_PHASE:
                signal_id = event["signal"]
                if signal_id not in oscillator_phase_cumulative:
                    oscillator_phase_cumulative[signal_id] = 0.0
                phase_incr = event["increment_oscillator_phase"]
                if isinstance(phase_incr, ParamRef):
                    try:
                        phase_incr = parameter_values[phase_incr.param_name]
                    except KeyError as e:
                        raise make_error_nt_param(phase_incr, event) from e
                oscillator_phase_cumulative[signal_id] += phase_incr
            if event["event_type"] == EventType.SET_OSCILLATOR_PHASE:
                signal_id = event["signal"]
                osc_phase = event["set_oscillator_phase"]
                if isinstance(osc_phase, ParamRef):
                    try:
                        osc_phase = parameter_values[osc_phase.param_name]
                    except KeyError as e:
                        raise make_error_nt_param(osc_phase, event) from e
                oscillator_phase_cumulative[signal_id] = osc_phase
                oscillator_phase_sets[signal_id] = event["time"]

            if event["event_type"] in [EventType.PLAY_START, EventType.ACQUIRE_START]:
                amplitude = event["amplitude"]
                if isinstance(amplitude, ParamRef):
                    _logger.debug(
                        "Resolving param name %s, parameter_values=%s",
                        amplitude.param_name,
                        parameter_values,
                    )
                    try:
                        amplitude = parameter_values[amplitude.param_name]
                    except KeyError as e:
                        raise make_error_nt_param(amplitude, event) from e
                    amplitude = (
                        round(amplitude * amplitude_resolution) / amplitude_resolution
                    )

                    self._event_graph.set_node_attributes(
                        event["id"], {"amplitude": amplitude}
                    )
                if amplitude is not None and abs(amplitude) > 1.0:
                    raise LabOneQException(
                        f"Magnitude of amplitude {amplitude} exceeding unity for event {event}"
                    )

                phase = event["phase"]
                if isinstance(phase, ParamRef):
                    try:
                        phase = parameter_values[phase.param_name]
                    except KeyError as e:
                        raise make_error_nt_param(phase, event) from e
                    self._event_graph.set_node_attributes(event["id"], {"phase": phase})

                oscillator_frequency = event["oscillator_frequency"]
                if isinstance(oscillator_frequency, ParamRef):
                    oscillator_frequency = parameter_values[
                        oscillator_frequency.param_name
                    ]
                    self._event_graph.set_node_attributes(
                        event["id"], {"oscillator_frequency": oscillator_frequency}
                    )

                play_pulse_parameters = event.get("play_pulse_parameters", {})
                pulse_pulse_parameters = event.get("pulse_pulse_parameters", {})
                has_updates = False
                for k, v in play_pulse_parameters.items():
                    if isinstance(v, ParamRef):
                        try:
                            play_pulse_parameters[k] = parameter_values[v.param_name]
                        except KeyError as e:
                            raise make_error_nt_param(v, event) from e
                        has_updates = True
                for k, v in pulse_pulse_parameters.items():
                    if isinstance(v, ParamRef):
                        try:
                            pulse_pulse_parameters[k] = parameter_values[v.param_name]
                        except KeyError as e:
                            raise make_error_nt_param(v, event) from e
                        has_updates = True
                if has_updates:
                    self._event_graph.set_node_attributes(
                        event["id"],
                        {
                            "play_pulse_parameters": play_pulse_parameters,
                            "pulse_pulse_parameters": pulse_pulse_parameters,
                        },
                    )

                oscillator_phase = None
                baseband_phase = None
                signal_id = event["signal"]
                signal_info = self._experiment_dao.signal_info(signal_id)
                oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
                if oscillator_info is not None:
                    if signal_info.modulation and signal_info.device_type in [
                        "hdawg",
                        "shfsg",
                    ]:
                        if oscillator_info.frequency_param is not None:
                            try:
                                frequency = parameter_values[
                                    oscillator_info.frequency_param
                                ]
                            except KeyError as e:
                                # HW oscillator sweeps with a NT parameter is fine - the
                                # controller will take care of it.
                                if not oscillator_info.hardware:
                                    raise make_error_nt_param(
                                        oscillator_info.frequency_param, event
                                    ) from e
                        else:
                            frequency = oscillator_info.frequency

                        incremented_phase = 0.0
                        if signal_id in oscillator_phase_cumulative:
                            incremented_phase = oscillator_phase_cumulative[signal_id]

                        if oscillator_info.hardware:
                            if len(oscillator_phase_sets) > 0:
                                raise Exception(
                                    f"There are set_oscillator_phase entries for signal '{signal_id}', but oscillator '{oscillator_info.id}' is a hardware oscillator. Setting absolute phase is not supported for hardware oscillators."
                                )
                            baseband_phase = incremented_phase
                        else:
                            phase_reference_time = phase_reset_time
                            if signal_id in oscillator_phase_sets:
                                phase_reference_time = max(
                                    phase_reset_time, oscillator_phase_sets[signal_id]
                                )
                            oscillator_phase = (
                                event["time"] - phase_reference_time
                            ) * 2.0 * math.pi * frequency + incremented_phase
                event_node = self._event_graph.node(event["id"])
                event_node["oscillator_phase"] = oscillator_phase
                event_node["baseband_phase"] = baseband_phase

    def _add_repetition_time_edges_for(self, section_name, repetition_time):
        if repetition_time is None:
            return

        section_info = self._section_graph_object.section_info(section_name)

        loop_step_start_events = self._event_graph.find_section_events_by_type(
            section_name, event_type=EventType.LOOP_STEP_START
        )

        loop_step_end_events = {
            (e["iteration"], e.get("loop_iteration")): e
            for e in self._event_graph.find_section_events_by_type(
                section_name, event_type=EventType.LOOP_STEP_END
            )
        }
        right_aligned_collector_events = {}
        if section_info.align == "right":
            right_aligned_collector_events = {}
            for iteration, e in loop_step_end_events.items():
                for edge in self._event_graph.out_edges(e["id"]):
                    other_node = self._event_graph.node(edge[1])
                    if (
                        other_node["event_type"] == EventType.RIGHT_ALIGNED_COLLECTOR
                        and other_node["section_name"] == section_name
                    ):
                        right_aligned_collector_events[iteration] = other_node

        for iteration_start_event in loop_step_start_events:
            key = (
                iteration_start_event["iteration"],
                iteration_start_event.get("loop_iteration"),
            )
            iteration_end_event = loop_step_end_events[key]
            if key in right_aligned_collector_events:
                iteration_end_event = right_aligned_collector_events[key]

            self._event_graph.after_exactly(
                iteration_end_event["id"], iteration_start_event["id"], repetition_time
            )

        section_span = self._event_graph.find_section_start_end(section_name)
        loop_iteration_end_event = next(
            e
            for e in self._event_graph.find_section_events_by_type(
                section_name, event_type=EventType.LOOP_ITERATION_END
            )
            if not e.get("shadow")
        )

        if section_info.align == "right":
            right_aligned_collector_of_iteration_end = None
            for edge in self._event_graph.out_edges(loop_iteration_end_event["id"]):
                other_node = self._event_graph.node(edge[1])
                if (
                    other_node["event_type"] == EventType.RIGHT_ALIGNED_COLLECTOR
                    and other_node["section_name"] == section_name
                ):
                    right_aligned_collector_of_iteration_end = other_node
            self._event_graph.after_at_least(
                right_aligned_collector_of_iteration_end["id"],
                section_span.start,
                repetition_time,
            )
        else:

            self._event_graph.after_at_least(
                loop_iteration_end_event["id"], section_span.start, repetition_time
            )

    def _find_repetition_mode_info(self, section):
        section_info = self._section_graph_object.section_info(section)
        if section_info.averaging_mode == "sequential" and (
            section_info.repetition_mode == "constant"
            or section_info.repetition_mode == "auto"
        ):
            return Box(
                {
                    k: getattr(section_info, k)
                    for k in [
                        "repetition_mode",
                        "repetition_time",
                        "averaging_mode",
                        "section_id",
                    ]
                }
            )

        has_repeating_child = False
        for child in self._section_graph_object.section_children(section):
            child_section_info = self._section_graph_object.section_info(child)
            if child_section_info.has_repeat:
                has_repeating_child = True

        if not has_repeating_child and (
            section_info.repetition_mode == "constant"
            or section_info.repetition_mode == "auto"
        ):
            return {
                k: getattr(section_info, k)
                for k in [
                    "repetition_mode",
                    "repetition_time",
                    "averaging_mode",
                    "section_id",
                ]
            }

        if not section_info.has_repeat:
            return None

        last_parent_section = None
        while True:
            parent_section = self._section_graph_object.parent(section)
            if parent_section is None or parent_section == last_parent_section:
                return None
            parent_section_info = self._section_graph_object.section_info(
                parent_section
            )
            if (
                parent_section_info.repetition_mode == "constant"
                or parent_section_info.repetition_mode == "auto"
            ) and parent_section_info.averaging_mode == "cyclic":

                return {
                    k: getattr(parent_section_info, k)
                    for k in [
                        "repetition_mode",
                        "repetition_time",
                        "averaging_mode",
                        "section_id",
                    ]
                }

            last_parent_section = parent_section

    def _add_repetition_time_edges(self):
        for section in self._section_graph_object.sections():
            repetition_mode_info = self._find_repetition_mode_info(section)
            if repetition_mode_info is not None:
                repetition_time = repetition_mode_info.get("repetition_time")
                if repetition_time is not None:
                    self._add_repetition_time_edges_for(section, repetition_time)

    def _add_feedback_time_edges(self):
        @dataclass
        class AcquiresMatches:
            acquire_node_id: Optional[int] = None
            match_node_ids: List[int] = field(default_factory=list)
            local: bool = False

        event_graph_modified = False
        acquires = OrderedDict()
        signals_to_handles = {}
        for event_id in self._sorted_events:
            event_data = self._event_graph.node(event_id)
            event_type = event_data["event_type"]
            if event_type == "ACQUIRE_START":
                handle = event_data["acquire_handle"]
                signal = event_data["signal"]
                signals_to_handles[signal] = handle
                acquires.setdefault(handle, []).append(AcquiresMatches())
            elif event_type == "ACQUIRE_END":
                try:
                    handle = signals_to_handles[event_data["signal"]]
                    last_acquire = acquires[handle][-1]
                    assert last_acquire.acquire_node_id is None
                    last_acquire.acquire_node_id = event_id
                except KeyError:
                    # Right aligned section - cannot schedule feedback. Error is handled
                    # later
                    pass
            elif event_type == "SECTION_START":
                handle = event_data.get("handle")
                if handle is not None:
                    try:
                        last_acquire = acquires[handle][-1]
                        last_acquire.match_node_ids.append(event_id)
                        last_acquire.local = event_data["local"]
                        if last_acquire.acquire_node_id is None:
                            # Right aligned section - cannot schedule feedback.
                            # Consider passing worst-case timing along for that case
                            raise LabOneQException(
                                "Could not compute match section timing "
                                "- end of acquire came before start. Did you use "
                                "right-alignment?"
                            )

                    except KeyError:
                        raise LabOneQException(
                            "No matching acquire found for handle %s; this "
                            "can also happen if a match section is responding to a "
                            "right-aligned acquire. "
                        )

        for acquires_per_handle in acquires.values():
            for acquire in acquires_per_handle:
                if acquire.match_node_ids:
                    event_graph_modified = True
                    for m in acquire.match_node_ids:
                        # N.B.: Careful with very short delays, they are difficult to
                        # represent with playZero/executeTableEntry; thus this delay
                        # will be at least be 32 samples long
                        #
                        if acquire.local:
                            # todo(JL): Proper timing model; but does it work for right alignment?
                            timing = lambda _: 50e-9
                        else:
                            # todo(JL): Proper timing model; but does it work for right alignment?
                            timing = lambda _: 500e-9
                        self._event_graph.after_at_least(
                            m, acquire.acquire_node_id, timing
                        )
        if event_graph_modified:
            self._sorted_events = self._event_graph.sorted_events()

    @dataclass
    class RepeatSectionsEntry:
        section_name: str
        num_repeats: int
        nesting_level: int
        loop_iteration_end_id: int
        loop_end_id: int
        parameter_list: List
        section_span: Any
        reset_phase_sw: bool
        reset_phase_hw: bool

    def add_repeat(
        self,
        section_name,
        num_repeats,
        parameter_list=None,
        reset_phase_sw=False,
        reset_phase_hw=False,
    ) -> RepeatSectionsEntry:
        if parameter_list is None:
            parameter_list = []
        _logger.debug("Adding repeat of %s", section_name)
        section_span = self._event_graph.find_section_start_end(section_name)
        nesting_level = len(self._path_to_root(section_name))

        loop_iteration_end_id = self._event_graph.add_node(
            section_name=section_name,
            event_type=EventType.LOOP_ITERATION_END,
            num_repeats=num_repeats,
            parameter_list=parameter_list,
            loop_start_node=section_span.start,
            nesting_level=nesting_level,
        )

        for edge in self._event_graph.out_edges(section_span.end):
            if edge[2]["relation"] == EventRelation.AFTER_OR_AT:
                self._event_graph.after_or_at(loop_iteration_end_id, edge[1])

        loop_end_id = self._event_graph.add_node(
            section_name=section_name,
            event_type=EventType.LOOP_END,
            num_repeats=num_repeats,
            nesting_level=nesting_level,
        )
        self._event_graph.add_edge(
            loop_end_id, loop_iteration_end_id, relation=EventRelation.AFTER_LOOP
        )

        self._event_graph.after_or_at(section_span.end, loop_end_id)

        return self.RepeatSectionsEntry(
            section_name,
            num_repeats,
            nesting_level,
            loop_iteration_end_id,
            loop_end_id,
            parameter_list,
            section_span,
            reset_phase_sw,
            reset_phase_hw,
        )

    def _nodes_which_reference_parameters(self):

        zero_getter = itemgetter(0)
        param_referencing_nodes = [
            (e[2].get("delta").param_name, e[0])
            for e in self._event_graph.edge_list()
            if isinstance(e[2].get("delta"), ParamRef)
        ]
        nodes_of_param_dict = {
            k: [node[1] for node in g]
            for k, g in groupby(
                sorted(param_referencing_nodes, key=zero_getter), zero_getter
            )
        }
        return nodes_of_param_dict

    def add_iteration_control_events(
        self, repeat_section_entry: RepeatSectionsEntry, first_only=False
    ):
        section_name = repeat_section_entry.section_name
        num_repeats = repeat_section_entry.num_repeats
        nesting_level = repeat_section_entry.nesting_level
        loop_iteration_end_id = repeat_section_entry.loop_iteration_end_id
        loop_end_id = repeat_section_entry.loop_end_id
        parameter_list = repeat_section_entry.parameter_list
        section_span = repeat_section_entry.section_span
        reset_phase_sw = repeat_section_entry.reset_phase_sw
        reset_phase_hw = repeat_section_entry.reset_phase_hw
        previous_loop_step_end_id = None

        repeats = num_repeats
        if first_only and num_repeats > 1:
            repeats = 1

        nodes_which_reference_params = self._nodes_which_reference_parameters()
        _logger.debug("Adding iteration events for %s", section_name)

        right_aligned = False
        if self._section_graph_object.section_info(section_name).align == "right":
            right_aligned = True

        for iteration in range(repeats):
            _logger.debug(
                "Processing iteration %d of num repeats= %d in section %s",
                iteration,
                num_repeats,
                section_name,
            )

            # Every step of the loop is delimited by 3 events: LOOP_STEP_START (LSS),
            # LOOP_STEP_BODY_START (LSBS) and LOOP_STEP_END (LSE).
            # LOOP_STEP_BODY_START marks the end of the loop preamble (P), and the start
            # of the actual body (B) of the loop. The preamble is used for setting
            # parameters like the oscillator frequency. Setting these may consume time,
            # so by having a dedicated time slot, we avoid them bleeding into
            # neighbouring sections.
            # The loop is enclosed between section_span.begin (SSB) and section_span.end
            # (SSE) and ended by LOOP_END (LE), and SUBSECTION_END (SSNE).
            # LOOP_ITERATION_END (LIE) marks the end of the first loop iteration
            # The body ends by loop_iteration_end_id; between this and the actual
            # LOOP_STEP_END, the preamble for the next step can be run
            #
            # SSB | LSS -P- LSBS -B- LSE LIE |: LSS -P- LSBS -B- LSE :| SSNE LE SSE

            loop_step_start_id = self._event_graph.add_node(
                section_name=section_name,
                event_type=EventType.LOOP_STEP_START,
                iteration=iteration,
                nesting_level=nesting_level,
            )

            loop_step_body_start_id = self._event_graph.add_node(
                section_name=section_name,
                event_type=EventType.LOOP_STEP_BODY_START,
                iteration=iteration,
                nesting_level=nesting_level,
            )

            loop_step_end_id = self._event_graph.add_node(
                section_name=section_name,
                event_type=EventType.LOOP_STEP_END,
                iteration=iteration,
                nesting_level=nesting_level,
            )

            if section_name not in self._loop_step_events:
                self._loop_step_events[section_name] = {}
            self._loop_step_events[section_name][iteration] = (
                loop_step_start_id,
                loop_step_body_start_id,
                loop_step_end_id,
            )

            preamble_inserter = _PreambleInserter(
                self._event_graph,
                previous_loop_step_end_id=previous_loop_step_end_id,
                before_loops_id=section_span.start,
                loop_step_start_id=loop_step_start_id,
            )

            if iteration == 0:
                # make sure the loop step end at least depends on the same nodes
                # as the loop iteration end
                for edge in self._event_graph.out_edges(loop_iteration_end_id):
                    if edge[2]["relation"] == EventRelation.AFTER_OR_AT:
                        self._event_graph.after_or_at(loop_step_end_id, edge[1])

                self._event_graph.after_or_at(loop_iteration_end_id, loop_step_end_id)

                subsection_events = [
                    event
                    for sublist in [
                        self._event_graph.find_section_events_by_type(
                            section_name, event_type
                        )
                        for event_type in (
                            EventType.SUBSECTION_START,
                            EventType.SUBSECTION_END,
                            EventType.PLAY_START,
                            EventType.PLAY_END,
                            EventType.DELAY_START,
                            EventType.DELAY_END,
                        )
                    ]
                    for event in sublist
                ]

                if right_aligned:
                    right_aligned_collector = next(
                        e
                        for e in self._event_graph.find_section_events_by_type(
                            section_name, EventType.RIGHT_ALIGNED_COLLECTOR
                        )
                    )["id"]
                    self._event_graph.after_or_at(
                        loop_step_end_id, right_aligned_collector
                    )

                nodes_linked_to_end = [
                    e[1] for e in self._event_graph.out_edges(section_span.end)
                ]

                nodes_linked_to_start = [
                    e[0] for e in self._event_graph.in_edges(section_span.start)
                ]

                for subsection_event in subsection_events:

                    subsection_event_id = subsection_event["id"]
                    if subsection_event_id in nodes_linked_to_end:
                        _logger.debug(
                            "Linking subsection event %s to loop_step_end_id=%s",
                            subsection_event,
                            loop_step_end_id,
                        )
                        self._event_graph.after_or_at(
                            loop_step_end_id, subsection_event["id"]
                        )
                    if subsection_event_id in nodes_linked_to_start:
                        self._event_graph.after_or_at(
                            subsection_event["id"], loop_step_start_id
                        )

            if iteration == 1:
                self._event_graph.after_or_at(loop_step_start_id, loop_iteration_end_id)

            if iteration == repeats - 1:
                self._event_graph.after_or_at(loop_end_id, loop_step_end_id)

            self._event_graph.after_or_at(loop_step_body_start_id, loop_step_start_id)
            self._event_graph.after_or_at(loop_step_end_id, loop_step_body_start_id)

            if previous_loop_step_end_id is not None:
                self._event_graph.after_or_at(
                    loop_step_start_id, previous_loop_step_end_id
                )
            else:
                self._event_graph.after_or_at(loop_step_start_id, section_span.start)

            if reset_phase_sw:
                reset_phase_sw_id = self._event_graph.add_node(
                    section_name=section_name,
                    event_type=EventType.RESET_SW_OSCILLATOR_PHASE,
                    iteration=iteration,
                )
                preamble_inserter.insert(
                    reset_phase_sw_id, add_before_first_loop_start=True
                )

            if reset_phase_hw:
                devices = {
                    self._experiment_dao.device_from_signal(s)
                    for s in self._experiment_dao.section_signals_with_children(
                        section_name
                    )
                }
                for device in devices:
                    device_info = self._experiment_dao.device_info(device)
                    try:
                        device_type = DeviceType(device_info.device_type)
                    except ValueError:
                        # Not every device has a corresponding DeviceType (e.g. PQSC)
                        continue
                    if not device_type.supports_reset_osc_phase:
                        continue
                    reset_phase_hw_id = self._event_graph.add_node(
                        section_name=section_name,
                        event_type=EventType.RESET_HW_OSCILLATOR_PHASE,
                        iteration=iteration,
                        duration=device_type.reset_osc_duration,
                        device_id=device_info.id,
                    )
                    preamble_inserter.insert(
                        reset_phase_hw_id, add_before_first_loop_start=False
                    )
                    self._event_graph.after_at_least(
                        loop_step_body_start_id,
                        reset_phase_hw_id,
                        device_type.reset_osc_duration,
                    )

            # Find oscillators driven by parameters
            # TODO(PW): for performance, consider moving out of loop over iterations
            oscillator_param_lookup = dict()
            for oscillator in self._experiment_dao.hardware_oscillators():
                oscillator_id = oscillator.id
                device_id = oscillator.device_id
                frequency_param = oscillator.frequency_param
                if frequency_param is None:
                    continue

                for signal_id in self._experiment_dao.signals():
                    oscillator = self._experiment_dao.signal_oscillator(signal_id)
                    # Not every signal has an oscillator (e.g. flux lines), so check for None
                    if oscillator is None:
                        continue
                    if (
                        frequency_param in oscillator_param_lookup
                        and oscillator_param_lookup[frequency_param]["id"]
                        != oscillator_id
                    ):
                        raise LabOneQException(
                            "Hardware frequency sweep may drive only a single oscillator"
                        )
                    if oscillator.id == oscillator_id:
                        oscillator_param_lookup[frequency_param] = {
                            "id": oscillator_id,
                            "device_id": device_id,
                            "signal_id": signal_id,
                        }

            for param in parameter_list:

                if param.get("values") is not None:
                    current_param_value = param["values"][iteration]
                else:
                    current_param_value = param["start"] + iteration * param["step"]

                param_set_id = self._event_graph.add_node(
                    section_name=section_name,
                    event_type=EventType.PARAMETER_SET,
                    parameter=param,
                    iteration=iteration,
                    value=current_param_value,
                )

                preamble_inserter.insert(param_set_id, add_before_first_loop_start=True)
                if iteration == 0:

                    param_referencing_nodes = nodes_which_reference_params.get(
                        param["id"], []
                    )
                    # Make sure that nodes which reference the parameter through parameterized edges
                    # are scheduled after the parameter setting node
                    for n in param_referencing_nodes:
                        self._event_graph.after_or_at(n, param_set_id)

                param_oscillator = oscillator_param_lookup.get(param["id"])
                if param_oscillator is not None:
                    osc_freq_start_id = self._event_graph.add_node(
                        section_name=section_name,
                        event_type=EventType.SET_OSCILLATOR_FREQUENCY_START,
                        parameter=param,
                        iteration=iteration,
                        value=current_param_value,
                        device_id=param_oscillator["device_id"],
                        signal=param_oscillator["signal_id"],
                    )

                    preamble_inserter.insert(
                        osc_freq_start_id, add_before_first_loop_start=True
                    )
                    device_id = param_oscillator["device_id"]
                    osc_freq_end_id = self._event_graph.add_node(
                        section_name=section_name,
                        event_type=EventType.SET_OSCILLATOR_FREQUENCY_END,
                        parameter=param,
                        iteration=iteration,
                        value=current_param_value,
                        device_id=device_id,
                        signal=param_oscillator["signal_id"],
                    )
                    device_type = DeviceType(
                        self._experiment_dao.device_info(device_id).device_type
                    )
                    oscillator_set_latency = max(
                        device_type.oscillator_set_latency,
                        device_type.min_play_wave / device_type.sampling_rate,
                    )
                    self._event_graph.after_at_least(
                        osc_freq_end_id, osc_freq_start_id, oscillator_set_latency
                    )
                    self._event_graph.after_or_at(
                        loop_step_body_start_id, osc_freq_end_id
                    )

            previous_loop_step_end_id = loop_step_end_id

        if previous_loop_step_end_id is not None:
            self._event_graph.after_or_at(section_span.end, previous_loop_step_end_id)
            self._event_graph.after_or_at(loop_end_id, previous_loop_step_end_id)

    def generate_loop_events(self, repeat_sections: Dict[str, RepeatSectionsEntry]):
        defined_parameters_of_children = set()
        for section_name in reversed(
            list(self._section_graph_object.topologically_sorted_sections())
        ):
            section_info = self._section_graph_object.section_info(section_name)
            if not section_info.has_repeat:
                _logger.debug("Section %s is not repeated", section_name)
                continue

            _logger.debug("Adding loop events for section %s", section_name)
            _logger.debug(section_info)

            iteration_events = self._event_graph.find_section_events_by_type(
                section_name, event_type=EventType.LOOP_ITERATION_END
            )
            _logger.debug("Iteration events:  %s", iteration_events)
            if len(iteration_events) > 1:
                _logger.warning("Mulitple iteration events found: %s", iteration_events)
            iteration_event = iteration_events[0]
            parameter_list = [
                param["id"] for param in iteration_event["parameter_list"]
            ]

            descendants = self._event_graph.descendants(iteration_event["id"])
            for descendant in descendants:
                event_data = self._event_graph.node(descendant)

                _logger.debug("Descendant:  %s", node_info(event_data))

            section_span = self._event_graph.find_section_start_end(section_name)
            section_start_descendants = self._event_graph.descendants(
                section_span.start
            )
            section_start_descendants.add(section_span.start)

            for descendant in section_start_descendants:
                event_data = self._event_graph.node(descendant)
                _logger.debug("Section start descendant:  %s", node_info(event_data))

            between_event_ids = set(descendants).difference(
                set(section_start_descendants)
            )

            filtered_between_event_ids = []
            for between_id in between_event_ids:
                event_data = self._event_graph.node(between_id)
                if (
                    event_data["event_type"] in ["LOOP_STEP_START", "LOOP_STEP_END"]
                    and event_data["section_name"] == section_name
                ):
                    _logger.debug("Filtering loop node:  %s", node_info(event_data))
                else:
                    filtered_between_event_ids.append(between_id)

            between_event_ids = filtered_between_event_ids

            self._event_graph.set_node_attributes(
                iteration_event["id"], {"events_in_iteration": list(between_event_ids)}
            )

            between_events = [
                self._event_graph.node(between_event_id)
                for between_event_id in between_event_ids
            ]

            has_parameterized_events = False
            referenced_parameters = set()
            defined_parameters = set()
            for event_data in between_events:
                _logger.debug("Between:  %s", node_info(event_data))
                if event_data["event_type"] == "PARAMETER_SET":
                    _logger.debug(
                        "There is a PARAMETER_SET event in between: %s", event_data
                    )
                    if event_data["section_name"] == section_name:
                        _logger.debug("And it is in this section %s", event_data)
                        has_parameterized_events = True
                    else:
                        _logger.debug("BUT it is not in this section %s", event_data)
                    param_obj = event_data["parameter"]
                    defined_parameters.add(param_obj["id"])
                if "parameterized_with" in event_data:
                    parameterized_with = event_data["parameterized_with"]
                    if parameterized_with is not None:
                        referenced_parameters.update(parameterized_with)

            missing_parameters = referenced_parameters.difference(set(parameter_list))
            missing_parameters = missing_parameters.difference(
                defined_parameters_of_children
            )
            _logger.debug(
                "Section %s has parameter_list %s and refers to parameters %s ,"
                + " missing parameters %s and defines parameters %s, children defined parameters %s",
                section_name,
                parameter_list,
                referenced_parameters,
                missing_parameters,
                defined_parameters,
                defined_parameters_of_children,
            )

            # todo (Pol): this is wrong. The previously visited node may have been a
            # sibling, not a child.
            defined_parameters_of_children = defined_parameters_of_children.union(
                set(parameter_list)
            )

            if (
                parameter_list == []
                and not has_parameterized_events
                and repeat_sections[section_name].num_repeats > 1
            ):
                _logger.debug("Compressing events of section %s", section_name)
                repeat_section = repeat_sections[section_name]
                self.add_iteration_control_events(repeat_section, first_only=True)
                _logger.debug(
                    "Setting compressed to true on node %s",
                    repeat_section.loop_iteration_end_id,
                )
                self._event_graph.set_node_attributes(
                    repeat_section.loop_iteration_end_id, {"compressed": True}
                )
                compressed = True
            else:
                _logger.debug("Not compressing events of section %s", section_name)

                self.add_iteration_control_events(repeat_sections[section_name])

                step_start_events = self._event_graph.find_section_events_by_type(
                    section_name, event_type=EventType.LOOP_STEP_START
                )
                _logger.debug(
                    "Found %d step start events for section %s",
                    len(step_start_events),
                    section_name,
                )

                self._event_graph.set_node_attributes(
                    iteration_event["id"], {"compressed": False}
                )
                compressed = False

            for (iteration, (_, step_body_id, step_end_id)) in self._loop_step_events[
                section_name
            ].items():

                if iteration == 0:
                    for event in between_events:
                        self._event_graph.after_or_at(event["id"], step_body_id)
                elif not compressed:
                    _logger.debug(
                        "Copying %d events for iteration %d",
                        len(between_events),
                        iteration,
                    )
                    self.copy_iteration_events(
                        between_events,
                        section_name,
                        iteration,
                        step_body_id,
                        step_end_id,
                    )

    def copy_iteration_events(
        self, events, section_name, iteration, step_body, step_end
    ):

        event_map = {}
        for event in events:
            if "section_name" not in event:
                raise Exception(f"No section name in {event}")
            new_id = self._event_graph.add_node(
                section_name=event["section_name"],
                event_type=event["event_type"],
                orig_event=event["id"],
                shadow=True,
                loop_iteration=section_name + "_" + str(iteration),
            )
            _logger.debug("Copying event %s to %s", event, new_id)

            attributes = {}
            for k, v in event.items():
                if k != "id":
                    attributes[k] = copy.deepcopy(v)
            self._event_graph.set_node_attributes(new_id, attributes)
            event_map[event["id"]] = new_id
            self._event_graph.after_or_at(new_id, step_body)
            self._event_graph.after_or_at(step_end, new_id)

        for event in events:
            new_node_id = event_map[event["id"]]
            new_event = self._event_graph.node(new_node_id)
            start_node_id = new_event.get("loop_start_node")
            if start_node_id in event_map:
                mapped_start_id = event_map[start_node_id]
                self._event_graph.set_node_attributes(
                    new_node_id, {"loop_start_node": mapped_start_id}
                )

            events_in_iteration = new_event.get("events_in_iteration")
            if events_in_iteration is not None:
                new_events_in_iteration = []
                for iteration_event_id in events_in_iteration:
                    if iteration_event_id in event_map:
                        new_events_in_iteration.append(event_map[iteration_event_id])
                    else:
                        new_events_in_iteration.append(iteration_event_id)
                self._event_graph.set_node_attributes(
                    new_node_id, {"events_in_iteration": new_events_in_iteration}
                )

            for edge in self._event_graph.out_edges(event["id"]):
                other_event = self._event_graph.node(edge[1])
                new_other_event = None
                if other_event["id"] in event_map:
                    new_other_event = event_map[other_event["id"]]
                if new_other_event is not None:
                    self._event_graph.add_edge(
                        new_node_id, new_other_event, relation=edge[2]["relation"]
                    )
                    for k, v in edge[2].items():
                        self._event_graph.set_edge_attribute(
                            new_node_id, new_other_event, k, v
                        )

    def add_play_wave_chain(
        self,
        play_wave_list: List[_PlayWave],
        parent_section_name,
        right_aligned=False,
        section_start_node=None,
    ):
        section_span = self._event_graph.find_section_start_end(parent_section_name)
        if section_start_node is None:
            section_start_node = section_span.start

        chain = []
        signal = None
        for i, play_wave in enumerate(play_wave_list):

            chain_element_id = parent_section_name + (play_wave.id or "") + str(i)
            default_attributes = {
                "section_name": parent_section_name,
                "signal": play_wave.signal,
            }
            if signal is None:
                signal = play_wave.signal
            else:
                if signal != play_wave.signal:
                    raise Exception(
                        f"Getting mixed signals: {signal} and {play_wave.signal}"
                    )

            if play_wave.precompensation_clear:
                if right_aligned:
                    # In principle, there is nothing that would stop us from
                    # implementing this. But the logic of the event graph is already
                    # quite complex, and PW doesn't want to strain it any more unless we
                    # actually need to.
                    raise LabOneQException(
                        "Precompensation clear not supported in right aligned sections"
                    )
                precompensation_clear_element = ChainElement(
                    id=chain_element_id + "RESET_PRECOMPENSATION_FILTERS_UNALIGNED",
                    start_type=EventType.RESET_PRECOMPENSATION_FILTERS_UNALIGNED,
                    end_type=None,
                    attributes={
                        **default_attributes,
                    },
                )
                chain.append(precompensation_clear_element)

            if play_wave.offset is not None:
                delay_element = ChainElement(
                    chain_element_id + "DELAY",
                    start_type=EventType.DELAY_START,
                    end_type=EventType.DELAY_END,
                    length=play_wave.offset,
                    attributes={**default_attributes, **{"play_wave_id": play_wave.id}},
                )
                chain.append(delay_element)

            if play_wave.increment_oscillator_phase is not None:

                increment_oscillator_phase_element = ChainElement(
                    id=chain_element_id + "INCREMENT_OSCILLATOR_PHASE",
                    start_type=EventType.INCREMENT_OSCILLATOR_PHASE,
                    end_type=None,
                    attributes={
                        **default_attributes,
                        **{
                            "increment_oscillator_phase": play_wave.increment_oscillator_phase
                        },
                    },
                )
                chain.append(increment_oscillator_phase_element)

            if play_wave.set_oscillator_phase is not None:

                set_oscillator_phase_element = ChainElement(
                    id=chain_element_id + "SET_OSCILLATOR_PHASE",
                    start_type=EventType.SET_OSCILLATOR_PHASE,
                    end_type=None,
                    attributes={
                        **default_attributes,
                        **{"set_oscillator_phase": play_wave.set_oscillator_phase},
                    },
                )
                chain.append(set_oscillator_phase_element)

            chain_element = ChainElement(
                chain_element_id,
                attributes={**default_attributes, **{"play_wave_id": play_wave.id}},
                start_attributes={},
                end_attributes={},
            )

            is_match_case_empty_section = (
                play_wave.play_wave_type == PlayWaveType.EMPTY_CASE
            )
            is_integration = play_wave.play_wave_type == PlayWaveType.INTEGRATION
            if (
                play_wave.length is not None
                or is_integration
                or is_match_case_empty_section
            ):
                chain_element.start_type = EventType.PLAY_START
                chain_element.end_type = EventType.PLAY_END
                if is_integration:
                    chain_element.start_type = EventType.ACQUIRE_START
                    chain_element.end_type = EventType.ACQUIRE_END
                elif play_wave.play_wave_type != PlayWaveType.PLAY:
                    chain_element.start_type = EventType.DELAY_START
                    chain_element.end_type = EventType.DELAY_END

                chain_element.length = play_wave.length
                chain_element.start_attributes["parameterized_with"] = [
                    p.param_name for p in play_wave.parameterized_with
                ]
                chain_element.start_attributes["phase"] = play_wave.phase
                chain_element.start_attributes["amplitude"] = play_wave.amplitude
                chain_element.start_attributes[
                    "oscillator_frequency"
                ] = play_wave.oscillator_frequency
                chain_element.start_attributes[
                    "acquire_handle"
                ] = play_wave.acquire_handle
                chain_element.start_attributes[
                    "acquisition_type"
                ] = play_wave.acquisition_type
                chain_element.start_attributes[
                    "play_wave_type"
                ] = play_wave.play_wave_type.name
                chain_element.end_attributes[
                    "play_wave_type"
                ] = play_wave.play_wave_type.name
                chain_element.start_attributes[
                    "play_pulse_parameters"
                ] = play_wave.play_pulse_parameters
                chain_element.start_attributes[
                    "pulse_pulse_parameters"
                ] = play_wave.pulse_pulse_parameters

                if play_wave.markers is not None:
                    chain_element.start_attributes["markers"] = [
                        copy.deepcopy(vars(m)) for m in play_wave.markers
                    ]

                if (
                    play_wave.play_wave_type != PlayWaveType.DELAY
                    or play_wave.length is not None
                    or is_match_case_empty_section
                ):
                    chain.append(chain_element)

        if right_aligned:
            right_aligned_collector_id = EventGraphBuilder.find_right_aligned_collector(
                self._event_graph, parent_section_name
            )
            if right_aligned_collector_id is None:
                raise Exception(
                    f"Found no RIGHT_ALIGNED_COLLECTOR in section {parent_section_name}"
                )
            EventGraphBuilder.add_right_aligned_chain(
                self._event_graph,
                section_start_node,
                right_aligned_collector_id,
                terminal_node_id=section_span.end,
                chain=chain,
            )

        else:
            terminal_id = section_span.end
            pull_out_node_id = None
            added_nodes = EventGraphBuilder.add_chain(
                self._event_graph,
                section_start_node,
                terminal_id,
                chain,
                reversed=False,
                link_last=True,
            )
            if pull_out_node_id is not None and len(chain) > 0:
                last_node_id = added_nodes[len(chain) - 1]["end_node_id"]
                self._event_graph.after_or_at(last_node_id, pull_out_node_id)

            for added_node in added_nodes.values():
                node_id = added_node["start_node_id"]
                node_data = self._event_graph.node(node_id)
                if (
                    node_data["event_type"]
                    == EventType.RESET_PRECOMPENSATION_FILTERS_UNALIGNED
                ):
                    aligned_node_id = self._event_graph.add_node(
                        event_type=EventType.RESET_PRECOMPENSATION_FILTERS,
                        section_name=node_data["section_name"],
                        signal_id=node_data["signal"],
                    )
                    EventGraphBuilder.add_time_link(
                        self._event_graph, aligned_node_id, node_id, None, False, False
                    )

    def process_events(self):
        _logger.debug("**** Event Timing")
        index = 0
        for event_id in self._event_timing.keys():
            event = self._event_graph.node(event_id)
            if event["event_type"] in [EventType.PLAY_START, EventType.PLAY_END]:
                event_time = event["time"]
                event_type = event["event_type"]
                signal = event["signal"]
                play_wave_id = event["play_wave_id"]
                time_beautified = EngNumber(float(event_time))
                if index < 500:
                    _logger.debug(
                        "%s %s %s %s", time_beautified, event_type, signal, play_wave_id
                    )
                else:
                    if index == 501:
                        _logger.debug("**Event log truncated")
                    _logger.debug(
                        "%s %s %s %s", time_beautified, event_type, signal, play_wave_id
                    )
            else:
                event_time = event["time"]
                event_type = event["event_type"]
                time_beautified = EngNumber(float(event_time))
                _logger.debug("%s %s", time_beautified, event_type)
            index += 1

    @staticmethod
    def _expand_loop_iterations(
        event_objects,
        loop_iteration_ends,
        loop_iteration_event,
        start_time,
        loop_iteration_lengths,
        offset,
        event_graph,
        max_events,
        events_added_ref,
        level=0,
    ):
        logheader = ""
        for i in range(level):
            logheader += "  "
        section_name = loop_iteration_event["section_name"]
        _logger.debug(
            "%sExpanding LOOP_ITERATION_END %s %s, at time %s, offset=%s",
            logheader,
            section_name,
            loop_iteration_event["id"],
            EngNumber(float(loop_iteration_event["time"])),
            EngNumber(float(offset)),
        )
        iteration_length = loop_iteration_lengths[section_name]
        iteration_start_time = loop_iteration_event["time"] - iteration_length

        _logger.debug(
            "%siteration_length=%s iteration_start_time=%s",
            logheader,
            EngNumber(float(iteration_length)),
            EngNumber(float(iteration_start_time)),
        )
        inner_logheader = logheader + " "
        expanded = 0
        for iteration in range(1, loop_iteration_event["num_repeats"]):

            _logger.debug(
                "%sExpanding iteration %d of iteration event %s at %s  section %s events_in_iteration=%s",
                logheader,
                iteration,
                loop_iteration_event["id"],
                EngNumber(float(loop_iteration_event["time"])),
                section_name,
                loop_iteration_event["events_in_iteration"],
            )

            for orig_event_id in loop_iteration_event["events_in_iteration"]:

                orig_event = event_graph.node(orig_event_id)
                shadow_event = copy.deepcopy(orig_event)
                shadow_event["shadow"] = True
                shadow_event["orig_event"] = orig_event_id
                shadow_event["iteration"] = iteration
                shadow_event["exp_iteration"] = iteration
                shadow_event["id"] = orig_event_id + 10000
                shadow_event["iterating_section_name"] = section_name
                shadow_event["iterating_event_id"] = loop_iteration_event["id"]

                relative_time = orig_event["time"] - iteration_start_time
                shadow_event["shadow_sequence"] = events_added_ref[0]

                new_time = (
                    iteration_length * iteration
                    + relative_time
                    + iteration_start_time
                    + offset
                )

                _logger.debug(
                    "%sCopying %s %s has time of %s and relative time of %s, start time is %s; offset is %f moving to %s",
                    inner_logheader,
                    orig_event["id"],
                    orig_event["event_type"],
                    EngNumber(float(orig_event["time"])),
                    EngNumber(float(relative_time)),
                    EngNumber(float(start_time)),
                    offset,
                    EngNumber(float(new_time)),
                )
                shadow_event["time"] = new_time
                event_objects.append(shadow_event)
                expanded += 1
                events_added_ref[0] += 1
                if max_events is not None and events_added_ref[0] >= max_events:
                    _logger.debug(
                        "%sTruncating events at %d", inner_logheader, max_events
                    )
                    break
                if (
                    orig_event["event_type"] == "LOOP_ITERATION_END"
                    and orig_event["section_name"] != section_name
                    and orig_event.get("compressed")
                ):

                    inner_offset = loop_iteration_event[
                        "time"
                    ] + loop_iteration_lengths[section_name] * (iteration - 1)
                    _logger.debug(
                        "%sDescending a level for iteration %d of iteration event %s  section %s",
                        inner_logheader,
                        iteration,
                        loop_iteration_event["id"],
                        section_name,
                    )
                    Scheduler._expand_loop_iterations(
                        event_objects,
                        loop_iteration_ends,
                        orig_event,
                        start_time=orig_event["time"]
                        - loop_iteration_lengths[orig_event["section_name"]],
                        loop_iteration_lengths=loop_iteration_lengths,
                        offset=inner_offset,
                        event_graph=event_graph,
                        max_events=max_events,
                        events_added_ref=events_added_ref,
                        level=level + 1,
                    )
                    if max_events is not None and events_added_ref[0] >= max_events:
                        _logger.debug(
                            "%sTruncating events at %d", inner_logheader, max_events
                        )
                        break
            if max_events is not None and events_added_ref[0] >= max_events:
                _logger.debug("%sTruncating events at %d", inner_logheader, max_events)
                break

        _logger.debug(
            "%sExpanded %d events for  LOOP_ITERATION_END %s",
            logheader,
            expanded,
            loop_iteration_event["id"],
        )

    def _events_ordered(self) -> Iterator[Any]:
        for event_id in self._event_timing.keys():
            event = self._event_graph.node(event_id)
            yield event

    @staticmethod
    def _calc_loop_iteration_info(events_ordered):
        loop_iteration_ends = {}
        loop_iteration_lengths = {}
        loop_starts = {}
        for event in events_ordered:
            section_name = event.get("section_name")
            if event["event_type"] == "LOOP_ITERATION_END":
                if section_name not in loop_iteration_ends:
                    loop_iteration_ends[section_name] = event
                    loop_iteration_lengths[section_name] = (
                        event["time"] - loop_starts[section_name]
                    )

            if event["event_type"] == "SECTION_START":
                existing_start = None
                if section_name in loop_starts:
                    existing_start = loop_starts[section_name]
                if existing_start is None or event["time"] < existing_start:
                    loop_starts[section_name] = event["time"]
        return loop_iteration_ends, loop_iteration_lengths, loop_starts

    def event_timing(self, expand_loops=True, max_events=None) -> List[Any]:

        events_added_ref = [0]
        (
            loop_iteration_ends,
            loop_iteration_lengths,
            loop_starts,
        ) = self._calc_loop_iteration_info(self._events_ordered())
        _logger.debug("Iteration lengths: %s", loop_iteration_lengths)
        too_short = set()
        for section, event in loop_iteration_ends.items():
            start_time = loop_starts[section]
            section_length = event["time"] - start_time
            section_devices = set()
            for signal in self._experiment_dao.section_signals_with_children(section):
                section_devices.add(
                    DeviceType(self._experiment_dao.signal_info(signal).device_type)
                )
            for device in section_devices:
                sampling_rate = device.sampling_rate
                if device == DeviceType.HDAWG:
                    sampling_rate = (
                        DeviceType.HDAWG.sampling_rate_2GHz
                        if self._clock_settings["use_2GHz_for_HDAWG"]
                        else DeviceType.HDAWG.sampling_rate
                    )
                section_length_samples = section_length * sampling_rate
                if section_length_samples < device.min_play_wave:
                    too_short.add(section)
        if not expand_loops:
            _logger.debug(
                "Sections %s are too short, will expand even though expand_loops is false",
                list(too_short),
            )

        event_objects: List[Any] = []

        for event in self._events_ordered():
            _logger.debug("event: %s", event)
            if (
                event["event_type"] == EventType.LOOP_ITERATION_END
                and "compressed" in event
                and event["compressed"]
                and (expand_loops or event["section_name"] in too_short)
            ):
                event_copy = copy.deepcopy(event)
                event_copy["compressed"] = False
                event_copy["expanded"] = True

                event_objects.append(event_copy)
                events_added_ref[0] += 1
                if max_events is not None and events_added_ref[0] >= max_events:
                    _logger.debug("Truncating events at %d", max_events)
                    break
                Scheduler._expand_loop_iterations(
                    event_objects,
                    loop_iteration_ends,
                    event,
                    loop_starts[event["section_name"]],
                    loop_iteration_lengths,
                    0,
                    self._event_graph,
                    max_events,
                    events_added_ref,
                )
            else:
                event_objects.append(copy.copy(event))
        for event_object in event_objects:
            event_object["accurate_time"] = repr(event_object["time"])
            event_object["time"] = float(event_object["time"])
            if "parameter" in event_object:
                event_object["parameter"] = {"id": event_object["parameter"].get("id")}
            if "parameter_list" in event_object:
                event_object["parameter_list"] = [
                    {"id": p.get("id")} for p in event_object["parameter_list"]
                ]

        event_objects = list(
            sorted(
                event_objects,
                key=lambda x: (
                    round(x.get("time") * 100e9),
                    x.get("exp_iteration") or 0,
                ),
            )
        )

        return event_objects

    def calculate_timing(self):
        (
            event_times,
            event_times_tiny_samples,
        ) = self._calculate_timing_for_graph(self._event_graph)

        repetition_mode_auto_sections = []
        if self._section_graph_object is not None:
            for section in self._section_graph_object.sections():
                repetition_mode_info = self._find_repetition_mode_info(section)
                if (
                    repetition_mode_info is not None
                    and repetition_mode_info.get("repetition_mode") == "auto"
                ):
                    repetition_mode_auto_sections.append(section)

            if len(repetition_mode_auto_sections) > 0:
                for repetition_mode_auto_section in repetition_mode_auto_sections:
                    loop_step_start_events = (
                        self._event_graph.find_section_events_by_type(
                            repetition_mode_auto_section,
                            event_type=EventType.LOOP_STEP_START,
                        )
                    )
                    loop_step_end_events = {
                        (e["iteration"], e.get("loop_iteration")): e
                        for e in self._event_graph.find_section_events_by_type(
                            repetition_mode_auto_section,
                            event_type=EventType.LOOP_STEP_END,
                        )
                    }
                    max_time = 0
                    max_iteration = None
                    for iteration_start_event in loop_step_start_events:
                        iteration_end_event = loop_step_end_events[
                            (
                                iteration_start_event["iteration"],
                                iteration_start_event.get("loop_iteration"),
                            )
                        ]
                        current_time = (
                            event_times_tiny_samples[iteration_end_event["id"]]
                            - event_times_tiny_samples[iteration_start_event["id"]]
                        )
                        if current_time > max_time:
                            max_time = current_time
                            max_iteration = iteration_end_event["id"]

                    section_span = self._event_graph.find_section_start_end(
                        repetition_mode_auto_section
                    )
                    loop_iteration_end_event = next(
                        e
                        for e in self._event_graph.find_section_events_by_type(
                            repetition_mode_auto_section,
                            event_type=EventType.LOOP_ITERATION_END,
                        )
                        if not e.get("shadow")
                    )
                    current_time = (
                        event_times_tiny_samples[loop_iteration_end_event["id"]]
                        - event_times_tiny_samples[section_span.start]
                    )
                    if current_time > max_time:
                        max_time = current_time
                        max_iteration = loop_iteration_end_event["id"]

                    max_time = max_time * self._settings.TINYSAMPLE
                    section_grid = self.section_grid(repetition_mode_auto_section)
                    max_time_raw = max_time
                    max_time = math.ceil(max_time / section_grid) * section_grid

                    _logger.debug(
                        "Max iteration time for section %s: %f, max_time_raw=%f max_iteration=%d",
                        repetition_mode_auto_section,
                        max_time,
                        max_time_raw,
                        max_iteration,
                    )
                    self._add_repetition_time_edges_for(
                        repetition_mode_auto_section, max_time
                    )
                _logger.debug(
                    "Performing second scheduling pass because of sections %s with repetition_mode auto",
                    repetition_mode_auto_sections,
                )
                (
                    event_times,
                    event_times_tiny_samples,
                ) = self._calculate_timing_for_graph(self._event_graph)

        for event_id, time in event_times.items():
            self._event_graph.node(event_id)["time"] = time

        self._event_timing = ValueSortedDict()
        sequence_nr = 0
        for event_id in self._sorted_events:
            event_data = self._event_graph.node(event_id)
            event_data["sequence_nr"] = sequence_nr
            event_time = float(event_data["time"])
            event_data["time"] = event_time
            self._event_timing[event_id] = (event_time, sequence_nr)
            sequence_nr += 1

        _logger.debug("Event times calculated")

    def _calculate_timing_for_graph(self, event_graph: EventGraph):
        event_times = {}
        parameter_change_times = SortedDict()

        def parameter_value(param_id, at_time):
            latest_value = None
            latest_time = None
            for k, v in parameter_change_times.items():
                if k[0] == param_id and at_time >= k[1]:
                    if latest_value is None or (
                        latest_time is None or k[1] >= latest_time
                    ):
                        latest_value = v
                        latest_time = k[1]
            return latest_value

        TINYSAMPLE = self._settings.TINYSAMPLE

        iteration_ends = {}

        for event_id in self._sorted_events:
            has_exactly_constraint = False

            event_data = event_graph.node(event_id)

            if (
                event_data["event_type"] == "LOOP_ITERATION_END"
                and "compressed" in event_data
                and event_data["compressed"]
            ):
                iteration_ends[event_data["section_name"]] = event_data

            event_time = 0
            earliest_event_time = 0
            latest_event_time = None

            if event_data["event_type"] in ["LOOP_END", "LOOP_STEP_END"]:
                if event_data["section_name"] in iteration_ends:
                    iteration_end = iteration_ends[event_data["section_name"]]
                    start_time = event_times[iteration_end["loop_start_node"]]

                    end_time = event_times[iteration_end["id"]]
                    iteration_length = end_time - start_time
                    if event_data["event_type"] == "LOOP_STEP_END":
                        loop_length = (event_data["iteration"] + 1) * iteration_length
                    else:
                        loop_length = iteration_end["num_repeats"] * iteration_length
                    loop_end_time = start_time + loop_length
                    earliest_event_time = max(earliest_event_time, loop_end_time)

            early_reference = None
            late_reference = None
            before_reference_time = None

            for _, other_event_id, edge in event_graph.out_edges(event_id):
                other_event = event_graph.node(other_event_id)
                other_time = event_times[other_event["id"]]
                if edge["relation"] == EventRelation.USES_EARLY_REFERENCE:
                    early_reference = other_time

                elif edge["relation"] == EventRelation.USES_LATE_REFERENCE:
                    late_reference = other_time

                elif edge["relation"] == EventRelation.AFTER_OR_AT:
                    event_time = max(other_time, event_time)
                    earliest_event_time = max(earliest_event_time, event_time)

                elif edge["relation"] == EventRelation.AFTER_AT_LEAST:
                    delay_time = edge["delta"]
                    if isinstance(delay_time, ParamRef):
                        delay_time = parameter_value(delay_time.param_name, other_time)
                    elif callable(delay_time):
                        delay_time = delay_time(other_time * TINYSAMPLE)

                    # todo (Pol): this causes off-by-one errors!
                    delay_time_tinysamples = math.floor(delay_time / TINYSAMPLE)
                    delayed_event_time = other_time + delay_time_tinysamples
                    earliest_event_time = max(earliest_event_time, delayed_event_time)
                    if delayed_event_time > event_time:
                        event_time = delayed_event_time

                elif edge["relation"] == EventRelation.AFTER_EXACTLY:
                    has_exactly_constraint = True
                    delay_time = edge["delta"]
                    if isinstance(delay_time, ParamRef):
                        delay_time = parameter_value(delay_time.param_name, other_time)
                    elif callable(delay_time):
                        delay_time = delay_time(other_time * TINYSAMPLE)
                    # todo (Pol): is this intended behavior?
                    delay_time_tinysamples = math.floor(delay_time / TINYSAMPLE)

                    delayed_event_time = other_time + delay_time_tinysamples
                    earliest_event_time = max(earliest_event_time, delayed_event_time)
                    if latest_event_time is None:
                        latest_event_time = delayed_event_time
                    else:
                        latest_event_time = min(delayed_event_time, latest_event_time)

                elif edge["relation"] == EventRelation.BEFORE_AT_LEAST:
                    delay_time = edge["delta"]
                    if isinstance(delay_time, ParamRef):
                        delay_time = parameter_value(delay_time.param_name, other_time)
                    elif callable(delay_time):
                        delay_time = delay_time(other_time * TINYSAMPLE)
                    # todo (Pol): is this intended behavior?
                    delayed_event_time = other_time - math.floor(
                        delay_time / TINYSAMPLE
                    )
                    if latest_event_time is None:
                        latest_event_time = delayed_event_time
                    else:
                        latest_event_time = min(delayed_event_time, latest_event_time)

                elif edge["relation"] == EventRelation.BEFORE_OR_AT:
                    if latest_event_time is not None:
                        latest_event_time = min(latest_event_time, other_time)
                    else:
                        latest_event_time = other_time

                elif edge["relation"] == EventRelation.RELATIVE_BEFORE:
                    before_reference_time = other_time

            if (
                early_reference is not None
                and late_reference is not None
                and before_reference_time is not None
            ):
                if latest_event_time is None:
                    latest_event_time = before_reference_time - (
                        late_reference - early_reference
                    )
                else:
                    latest_event_time = min(
                        latest_event_time,
                        before_reference_time - (late_reference - early_reference),
                    )

            event_type = event_data["event_type"]
            if event_type == EventType.SKELETON:
                event_type = event_data["skeleton_of"]

            if event_type in [
                EventType.SECTION_END,
                EventType.SECTION_START,
                EventType.RIGHT_ALIGNED_COLLECTOR,
                EventType.SPECTROSCOPY_END,
                EventType.LOOP_ITERATION_END,
                EventType.LOOP_STEP_BODY_START,
                EventType.LOOP_STEP_END,
                EventType.SUBSECTION_START,
                EventType.SUBSECTION_END,
                EventType.SECTION_SKELETON,
                EventType.RESET_PRECOMPENSATION_FILTERS,
                EventType.DIGITAL_SIGNAL_STATE_CHANGE,
            ]:
                if (
                    event_type
                    in [
                        EventType.SUBSECTION_START,
                        EventType.SUBSECTION_END,
                        EventType.SECTION_SKELETON,
                    ]
                    and "subsection_name" in event_data
                ):
                    # subsection determines the relevant grid for subsection start events
                    event_section = event_data["subsection_name"]
                else:
                    event_section = event_data["section_name"]

                section_grid = self.section_grid(event_section)
                section_grid_in_tinysamples = round(section_grid / TINYSAMPLE)

                corrected__earliest_event_time = (
                    (earliest_event_time + section_grid_in_tinysamples - 1)
                    // section_grid_in_tinysamples
                    * section_grid_in_tinysamples
                )

                if latest_event_time is not None:
                    corrected__latest_event_time = (
                        (latest_event_time + section_grid_in_tinysamples - 1)
                        // section_grid_in_tinysamples
                        * section_grid_in_tinysamples
                    )
                    if (
                        not has_exactly_constraint
                        and corrected__latest_event_time > latest_event_time
                    ):
                        corrected__latest_event_time -= section_grid_in_tinysamples
                else:
                    corrected__latest_event_time = latest_event_time

            else:
                corrected__earliest_event_time = earliest_event_time
                corrected__latest_event_time = latest_event_time

            if (
                corrected__latest_event_time is not None
                and corrected__earliest_event_time - corrected__latest_event_time
                > self._settings.CONSTRAINT_TOLERANCE / TINYSAMPLE
            ):
                exactly_constraint = None
                _logger.warning("Event %s %s", event_data["event_type"], event_data)
                for edge in event_graph.out_edges(event_id):
                    node1 = event_graph.node(edge[0])
                    node2 = event_graph.node(edge[1])
                    deltatext = ""
                    if edge[2].get("delta") is not None:
                        deltatext = f" delta={edge[2].get('delta')}"
                    _logger.warning(
                        "  Edge  %s %s %s -> %s at time %s",
                        edge[2]["relation"].name,
                        deltatext,
                        node_info(node1),
                        node_info(node2),
                        event_times[edge[1]],
                    )
                    if edge[2]["relation"] == EventRelation.AFTER_EXACTLY:
                        exactly_constraint = {
                            "other_node": edge[0],
                            "edge": edge,
                            "delta": edge[2].get("delta"),
                        }
                _logger.warning(
                    "corrected__earliest_event_time %f > corrected__latest_event_time %f",
                    corrected__earliest_event_time,
                    corrected__latest_event_time,
                )

                if (
                    event_type
                    in [EventType.SECTION_END, EventType.RIGHT_ALIGNED_COLLECTOR]
                    or (
                        event_type == EventType.SKELETON
                        and event_data.get("skeleton_of") == EventType.SECTION_END
                    )
                ) and exactly_constraint is not None:
                    raise LabOneQException(
                        f"Section end of section {event_data['section_name'] } must "
                        f"happen exactly {exactly_constraint['delta']} s after its "
                        f"start, but it contains operations which take longer."
                    )
                if (
                    event_type in [EventType.LOOP_STEP_END]
                ) and exactly_constraint is not None:
                    iteration = event_data.get("iteration")
                    raise LabOneQException(
                        f"Sweep step end of iteration {iteration} in section "
                        f"{event_data['section_name'] }  must happen exactly "
                        f"{exactly_constraint['delta']} s after its start, but it "
                        f"contains operations which take longer."
                    )

                raise Exception(
                    f"Inconsistent constraints for event {node_info(node1)}"
                )
            else:
                if corrected__latest_event_time is not None:
                    event_time = corrected__latest_event_time
                else:
                    event_time = corrected__earliest_event_time

            _logger.debug(
                "Calculated event time for event %s to be %f tinysamples",
                event_id,
                event_time,
            )
            event_times[event_id] = event_time

            if event_data["event_type"] == EventType.PARAMETER_SET:
                param_obj = event_data["parameter"]
                parameter_change_times[(param_obj["id"], event_time)] = event_data[
                    "value"
                ]

        event_times_tiny_samples = copy.deepcopy(event_times)

        for k in event_times.keys():
            event_times[k] = event_times[k] * TINYSAMPLE

        return event_times, event_times_tiny_samples

    def verify_timing(self):
        TOLERANCE = 1e-11
        # find nodes which are the reference for another node
        # through a parameterized edge indicating an "after" relation
        # this results in the earlier node of the relation being in the list, they are the
        # second item (b) in the edge (a,b), "a AFTER b"
        # we need the earlier nodes here so that a new parameter setting
        # has not yet overridden the current parameter value
        raw_param_referencing_nodes = [
            (e[1], e[2].get("delta").param_name)
            for e in self._event_graph.edge_list()
            if isinstance(e[2].get("delta"), ParamRef)
            and e[2].get("relation")
            in {EventRelation.AFTER_AT_LEAST, EventRelation.AFTER_EXACTLY}
        ]
        # add the nodes which are the reference for another node
        # through a parameterized edge indicating a "before" relation
        # this also results in the earlier node ending up in the list, they are the
        # first item (a) in the edge (a,b), "a BEFORE b"
        raw_param_referencing_nodes.extend(
            [
                (e[0], e[2].get("delta").param_name)
                for e in self._event_graph.edge_list()
                if isinstance(e[2].get("delta"), ParamRef)
                and e[2].get("relation") in {EventRelation.BEFORE_AT_LEAST}
            ]
        )

        zerogetter = itemgetter(0)
        # a dict with a node id as a key, mapping to a list of parameters
        # that are referenced through a parameterized edge by the key node
        parameter_referencing_nodes = {
            k: set([node[1] for node in g])
            for k, g in groupby(
                sorted(raw_param_referencing_nodes, key=zerogetter), zerogetter
            )
        }

        def add_issue(
            relation, event_time, other_time, delta, event, other_event, edge_data
        ):
            issues.append(
                {
                    "relation": relation,
                    "event_time": event_time,
                    "other_time": other_time,
                    "delta": delta,
                    "event": event,
                    "other_event": other_event,
                    "edge_data": edge_data,
                }
            )

        parameter_values = {}
        parameter_values_at_node = {}
        sorted_events = sorted(
            [
                self._event_graph.node(event_id)
                for event_id in self._event_graph.node_ids()
            ],
            # PARAMETER_SET must be visited early in case of tie
            key=lambda node: (node["time"], node["event_type"] != "PARAMETER_SET"),
        )
        issues = []

        for event in sorted_events:
            if event["event_type"] == EventType.PARAMETER_SET:
                parameter_values[event["parameter"]["id"]] = event["value"]

            if event["id"] in parameter_referencing_nodes:
                parameter_values_at_node[event["id"]] = {
                    k: parameter_values[k]
                    for k in parameter_referencing_nodes[event["id"]]
                }

            for _, other_node_id, edge_data in self._event_graph.out_edges(event["id"]):

                other_event = self._event_graph.node(other_node_id)
                relation = edge_data["relation"]

                if "delta" in edge_data:
                    if isinstance(edge_data["delta"], ParamRef):
                        param_name = edge_data["delta"].param_name
                        # get the value of the parameter as it was at the node at the other end of the edge
                        # because if a parameter change happens at the same time as the end of the interval
                        # the new value would be used - which is wrong because that value applies only to the
                        # next interval
                        param_values_at_other_node = parameter_values_at_node.get(
                            other_node_id
                        )
                        if param_values_at_other_node is not None:
                            delta = param_values_at_other_node[param_name]
                        else:
                            delta = parameter_values[param_name]

                    elif callable(edge_data["delta"]):
                        delta = edge_data["delta"](other_event["time"])
                    else:
                        delta = edge_data["delta"]
                else:
                    delta = None

                if (
                    "SKELETON" in event["event_type"]
                    or "SKELETON" in other_event["event_type"]
                ):
                    # SKELETON events have a difficult parameter-dependent timing -> ignore
                    continue

                if relation == EventRelation.AFTER_OR_AT:
                    if not event["time"] >= other_event["time"]:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )

                elif relation == EventRelation.AFTER_AT_LEAST:
                    if callable(delta):
                        delta = delta(other_event["time"])
                    if not event["time"] >= other_event["time"] + delta - TOLERANCE:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )
                elif relation == EventRelation.AFTER:
                    if not event["time"] > other_event["time"]:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )

                elif relation in (
                    EventRelation.USES_EARLY_REFERENCE,
                    EventRelation.USES_LATE_REFERENCE,
                ):
                    pass
                elif relation == EventRelation.AFTER_EXACTLY and delta is not None:
                    if not event["time"] - other_event["time"] - delta >= -TOLERANCE:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )

                elif relation == EventRelation.AFTER_LOOP:
                    pass
                    # assert event["time"] == other_event["time"]
                elif relation == EventRelation.BEFORE:
                    if not event["time"] < other_event["time"]:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )

                elif relation == EventRelation.BEFORE_OR_AT:
                    if not event["time"] <= other_event["time"]:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )

                elif relation == EventRelation.BEFORE_AT_LEAST:
                    if not event["time"] + delta <= other_event["time"] + TOLERANCE:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )

                elif relation == EventRelation.RELATIVE_BEFORE:
                    if not event["time"] <= other_event["time"]:
                        add_issue(
                            relation,
                            event["time"],
                            other_event["time"],
                            delta,
                            event,
                            other_event,
                            edge_data,
                        )
                else:
                    raise ValueError("invalid relation")

        for issue in issues:
            issue_text = ", ".join(f"{key}: {value}" for key, value in issue.items())
            _logger.warning(
                f"Issue: Not fulfilled timing relation {issue['event']['id']}  ->  {issue['other_event']['id']} : {issue_text}"
            )
        if len(issues) > 0 and not self._settings.IGNORE_GRAPH_VERIFY_RESULTS:
            raise LabOneQException(
                f"Scheduler issues: Constraints not satisfied: {issue_text}"
            )

    def _has_control_elements(self, section_display_name):
        section_info = self._experiment_dao.section_info(section_display_name)

        if (
            section_info.has_repeat
            or section_info.acquisition_types is not None
            and len(section_info.acquisition_types) > 0
            or section_info.handle is not None
            or section_info.state is not None
        ):
            return True
        for trigger_info in section_info.trigger_output:
            if trigger_info["state"]:
                return True

        for signal in self._experiment_dao.section_signals(section_display_name):
            if any(
                pulse.precompensation_clear
                for pulse in self._experiment_dao.section_pulses(
                    section_display_name, signal
                )
            ):
                return True

        return False

    def section_grid(self, section):

        if section in self._section_grids:
            _logger.debug("Section cache hit for section %s", section)
            return self._section_grids[section]

        if self._settings.FIXED_SLOW_SECTION_GRID:
            section_grid = Fraction(8, int(600e6))
            _logger.warning(
                "Fixed section grid because setting 'fixed_slow_section_grid' is set"
            )
            self._section_grids[section] = section_grid
            return section_grid

        if self._section_graph_object is not None:
            section_info = self._section_graph_object.section_info(section)
            section_display_name = section_info.section_display_name
        else:
            section_info = self._experiment_dao.section_info(section)
            section_display_name = section

        device_types = [
            DeviceType(device_type)
            for device_type in self._experiment_dao.device_types_in_section(
                section_display_name
            )
        ]
        has_control_elements = self._has_control_elements(section_display_name)

        _logger.debug("Device types for section %s are %s", section, device_types)
        self._section_grids[section] = self._calculate_section_grid(
            section,
            device_types,
            self._clock_settings["use_2GHz_for_HDAWG"],
            has_control_elements,
            section_info.on_system_grid,
        )
        return self._section_grids[section]

    @staticmethod
    def _calculate_section_grid(
        section,
        device_types,
        use_2GHz_for_HDAWG,
        has_control_elements,
        on_system_grid=False,
    ):

        if len(device_types) == 0:
            _logger.debug(
                "Section %s : using default section grid, device_types=%s",
                section,
                device_types,
            )
            return Fraction(8, 600 * 1000 * 1000)

        sequencer_frequencies = []
        signal_frequencies = set()
        for d in device_types:
            if d == DeviceType.HDAWG and use_2GHz_for_HDAWG:
                sequencer_frequencies.append(
                    int(d.sampling_rate_2GHz / d.sample_multiple)
                )
                signal_frequencies.add(d.sampling_rate_2GHz)
            else:
                sequencer_frequencies.append(int(d.sampling_rate / d.sample_multiple))
                signal_frequencies.add(d.sampling_rate)

        signal_frequencies = list(signal_frequencies)
        if (
            len(signal_frequencies) == 1
            and not has_control_elements
            and not on_system_grid
        ):
            _logger.debug(
                "signal frequencies are %s - returning signal frequency grid",
                [str(EngNumber(f)) for f in signal_frequencies],
            )
            return 1 / signal_frequencies[0]

        # Actually, we should return the total system grid when on_system_grid is True
        # and not only the sequencer frequencies here, but since this scheduler will
        # be replaced soon and the two grids anyway match for the relevant case, this is
        # fine.
        _logger.debug(
            "Sequencer frequencies are %s",
            [str(EngNumber(f)) for f in sequencer_frequencies],
        )
        sequencer_frequency = np.gcd.reduce(sequencer_frequencies)
        return 1 / Fraction(int(sequencer_frequency))

    def _add_start_events(self):
        retval = []

        # Add initial events to reset the NCOs.
        # Todo (PW): Drop once system tests have been migrated from legacy behaviour.
        for device_info in self._experiment_dao.device_infos():
            try:
                device_type = DeviceType(device_info.device_type)
            except ValueError:
                # Not every device has a corresponding DeviceType (e.g. PQSC)
                continue
            if not device_type.supports_reset_osc_phase:
                continue
            retval.append(
                self._event_graph.add_node(
                    event_type=EventType.INITIAL_RESET_HW_OSCILLATOR_PHASE,
                    device_id=device_info.id,
                    duration=device_type.reset_osc_duration,
                )
            )
        return retval

    def _path_to_root(self, section_id):
        current_section = section_id
        path_to_root = [current_section]
        while True:
            parent = self._experiment_dao.section_parent(current_section)
            if parent is not None:
                path_to_root.append(parent)
                current_section = parent
            else:
                break
        _logger.debug("Path to root from %s : %s", section_id, path_to_root)
        return path_to_root
