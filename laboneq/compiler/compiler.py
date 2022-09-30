# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field, asdict
import logging
import math
import json
import csv
import copy
import os
from collections import Counter, namedtuple

import numpy as np
from fractions import Fraction
from typing import Any, Iterator, List, Dict, Set, Tuple, Union, Optional
from operator import itemgetter
from sortedcollections import ValueSortedDict, SortedDict
from pathlib import Path
from engineering_notation import EngNumber

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.core.types.enums.acquisition_type import AcquisitionType

from .code_generator import (
    AWGSignalType,
    CodeGenerator,
    AWGInfo,
    SignalObj,
    TriggerMode,
)
from .compiler_settings import CompilerSettings
from .recipe_generator import RecipeGenerator
from .experiment_dao import ExperimentDAO
from .event_graph import EventGraph, EventRelation, EventType
from .fastlogging import NullLogger
from .device_type import DeviceType
from .event_graph_builder import EventGraphBuilder, ChainElement
from .section_graph import SectionGraph
from .measurement_calculator import MeasurementCalculator
from laboneq.core.exceptions import LabOneQException

_logger = logging.getLogger(__name__)
_dlogger = None
_dlog = False


@dataclass
class ParamRef:
    param_name: str


@dataclass
class _PlayWave:
    id: str
    signal: str
    length: float = None
    offset: Any = None
    amplitude: Any = None
    is_integration: bool = False
    signal_offset: Any = None
    parameterized_with: list = field(default_factory=list)
    acquire_handle: str = None
    acquisition_type: list = field(default_factory=list)
    phase: float = None
    increment_oscillator_phase: float = None
    set_oscillator_phase: float = None
    is_delay: bool = False


@dataclass
class LeaderProperties:
    global_leader: str = None
    is_desktop_setup: bool = False
    internal_followers: List[str] = field(default_factory=list)


_AWGMapping = Dict[str, Dict[int, AWGInfo]]


def _calculate_compiler_settings(local_settings: Optional[Dict] = None):
    def to_value(input_string):
        try:
            return int(input_string)
        except ValueError:
            pass
        try:
            return float(input_string)
        except ValueError:
            pass
        if input_string.lower() in ["true", "false"]:
            return input_string.lower() == "true"

    PREFIX = "QCCS_COMPILER_"
    compiler_settings_dict = asdict(CompilerSettings())

    for settings_key in compiler_settings_dict.keys():
        key = PREFIX + settings_key
        if key in os.environ:
            value = to_value(os.environ[key])
            if value is not None:
                compiler_settings_dict[settings_key] = value
                _logger.warning(
                    "Environment variable %s is set. %s overridden to be %s instead of default value %s",
                    key,
                    settings_key,
                    value,
                    getattr(CompilerSettings, settings_key),
                )
        else:
            _logger.debug("Key %s not found in environment variables", key)

    if local_settings is not None:
        for k, v in local_settings.items():
            if not k in compiler_settings_dict:
                raise KeyError(f"Not a valid setting: {k}")
            compiler_settings_dict[k] = v

    compiler_settings = CompilerSettings(**compiler_settings_dict)

    for k, v in asdict(compiler_settings).items():
        _logger.debug("Setting %s=%s", k, v)

    return compiler_settings


def _set_playwave_value_or_amplitude(section_pulse, play_wave, name, value=None):
    if value is not None:
        setattr(play_wave, name, value)
    elif section_pulse[name] is not None:
        setattr(play_wave, name, section_pulse[name])
    if section_pulse[name + "_param"] is not None:
        param_ref = ParamRef(section_pulse[name + "_param"])
        setattr(play_wave, name, param_ref)
        play_wave.parameterized_with.append(param_ref)


class Compiler:
    def __init__(self, settings: Optional[Dict] = None):

        global _dlogger, _dlog
        if _logger.getEffectiveLevel() == logging.DEBUG:
            _logger.info("Additional debug logging enabled for %s", __name__)
            _dlogger = _logger
            _dlog = True
        else:
            _logger.info("Additional debug logging disabled for %s", __name__)
            _dlogger = NullLogger()
            _dlog = False

        self._event_graph = EventGraph()
        self._osc_numbering = None
        self._section_grids = {}
        self._experiment_dao: ExperimentDAO = None
        self._settings = _calculate_compiler_settings(settings)
        self._section_events = {}

        self._loop_step_events: Dict[str, Dict[int, Tuple[int, int, int]]] = {}
        self._event_timing_compressed = {}
        self._leader_properties = LeaderProperties()
        self._clock_settings = {}
        self._integration_unit_allocation = None
        self._section_graph_object = None
        self._sampling_rate_cache = {}
        self._awgs: _AWGMapping = {}

        _logger.info("Starting QCCS Compiler run...")
        self._check_tinysamples()

    def _check_tinysamples(self):
        for t in DeviceType:
            num_tinysamples_per_sample = (
                1 / t.sampling_rate
            ) / self._settings.TINYSAMPLE
            delta = abs(round(num_tinysamples_per_sample) - num_tinysamples_per_sample)
            if delta > 1e-11:
                raise RuntimeError(
                    f"TINYSAMPLE is not commensurable with sampling rate of {t}, has {num_tinysamples_per_sample} tinysamples per sample, which is not an integer"
                )

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
        if section_info.get("align") == "right":
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

        if section_info.get("align") == "right":
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

    def _add_repetition_time_edges(self):
        for section in self._section_graph_object.sections():
            repetition_mode_info = self._find_repetition_mode_info(section)
            if repetition_mode_info is not None:
                repetition_time = repetition_mode_info.get("repetition_time")
                if repetition_time is not None:
                    self._add_repetition_time_edges_for(section, repetition_time)

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

    def add_iteration_control_events(
        self, repeat_section_entry: RepeatSectionsEntry, first_only=False,
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
        previous_iteration_end_id = None

        repeats = num_repeats
        if first_only and num_repeats > 1:
            repeats = 1

        _logger.debug("Adding iteration events for %s", section_name)

        right_aligned = False
        if self._section_graph_object.section_info(section_name)["align"] == "right":
            right_aligned = True

        for iteration in range(repeats):
            _dlogger.debug(
                "Processing iteration %d of num repeats= %d in section %s",
                iteration,
                num_repeats,
                section_name,
            )

            # Every step of the loop is delimited by 3 events: LOOP_STEP_START,
            # LOOP_STEP_BODY_START and LOOP_STEP_END.
            # LOOP_STEP_BODY_START marks the end of the loop preamble, and the start
            # of the actual body of the loop. The preamble is used for setting
            # parameters like the oscillator frequency. Setting these may consume time,
            # so by having a dedicated time slot, we avoid them bleeding into
            # neighbouring sections.

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

            if iteration == 0:
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
                        _dlogger.debug(
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

            if previous_iteration_end_id is not None:
                self._event_graph.after_or_at(
                    loop_step_start_id, previous_iteration_end_id
                )
            else:
                self._event_graph.after_or_at(loop_step_start_id, section_span.start)

            if reset_phase_sw:
                reset_phase_sw_id = self._event_graph.add_node(
                    section_name=section_name,
                    event_type=EventType.RESET_SW_OSCILLATOR_PHASE,
                    iteration=iteration,
                )
                if previous_iteration_end_id is not None:
                    self._event_graph.after_or_at(
                        reset_phase_sw_id, previous_iteration_end_id
                    )
                else:
                    self._event_graph.after_or_at(section_span.start, reset_phase_sw_id)
                self._event_graph.after_or_at(loop_step_start_id, reset_phase_sw_id)

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
                        device_type = DeviceType(device_info["device_type"])
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
                        device_id=device_info["id"],
                    )
                    if previous_iteration_end_id is not None:
                        self._event_graph.after_or_at(
                            reset_phase_hw_id, previous_iteration_end_id
                        )
                    else:
                        self._event_graph.after_or_at(
                            reset_phase_hw_id, section_span.start
                        )
                    self._event_graph.after_or_at(loop_step_start_id, reset_phase_hw_id)
                    self._event_graph.after_at_least(
                        loop_step_body_start_id,
                        reset_phase_hw_id,
                        device_type.reset_osc_duration,
                    )

            # Find oscillators driven by parameters
            # TODO(PW): for performance, consider moving out of loop over iterations
            oscillator_param_lookup = dict()
            for oscillator in self._experiment_dao.hardware_oscillators():
                oscillator_id = oscillator.get("id")
                device_id = oscillator.get("device_id")
                frequency_param = oscillator.get("frequency_param")
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
                    if oscillator["id"] == oscillator_id:
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

                if previous_iteration_end_id is not None:
                    self._event_graph.after_or_at(
                        param_set_id, previous_iteration_end_id
                    )
                else:
                    self._event_graph.after_or_at(section_span.start, param_set_id)
                self._event_graph.after_or_at(loop_step_start_id, param_set_id)

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

                    if previous_iteration_end_id is not None:
                        self._event_graph.after_or_at(
                            osc_freq_start_id, previous_iteration_end_id
                        )
                    else:
                        self._event_graph.after_or_at(
                            section_span.start, osc_freq_start_id
                        )
                    self._event_graph.after_or_at(loop_step_start_id, osc_freq_start_id)
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
                        self._experiment_dao.device_info(device_id)["device_type"]
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

            previous_iteration_end_id = loop_step_end_id

        if previous_iteration_end_id is not None:
            self._event_graph.after_or_at(section_span.end, previous_iteration_end_id)
            self._event_graph.after_or_at(loop_end_id, previous_iteration_end_id)

    def generate_loop_events(
        self, repeat_sections: Dict[str, "Compiler.RepeatSectionsEntry"]
    ):
        defined_parameters_of_children = set()
        for section_name in list(
            reversed(list(self._section_graph_object.topologically_sorted_sections()))
        ):
            section_info = self._section_graph_object.section_info(section_name)
            if not section_info["has_repeat"]:
                _dlogger.debug("Section %s is not repeated", section_name)
                continue

            _dlogger.debug("Adding loop events for section %s", section_name)
            _dlogger.debug(section_info)

            iteration_events = self._event_graph.find_section_events_by_type(
                section_name, event_type=EventType.LOOP_ITERATION_END
            )
            _dlogger.debug("Iteration events:  %s", iteration_events)
            if len(iteration_events) > 1:
                _logger.warning("Mulitple iteration events found: %s", iteration_events)
            iteration_event = iteration_events[0]
            parameter_list = list(
                map(itemgetter("id"), iteration_event["parameter_list"])
            )

            descendants = self._event_graph.descendants(iteration_event["id"])
            for descendant in descendants:
                event_data = self._event_graph.node(descendant)

                _dlogger.debug("Descendant:  %s", node_info(event_data))

            section_span = self._event_graph.find_section_start_end(section_name)
            section_start_descendants = self._event_graph.descendants(
                section_span.start
            )
            section_start_descendants.add(section_span.start)

            for descendant in section_start_descendants:
                event_data = self._event_graph.node(descendant)
                _dlogger.debug("Section start descendant:  %s", node_info(event_data))

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
                    _dlogger.debug("Filtering loop node:  %s", node_info(event_data))
                else:
                    filtered_between_event_ids.append(between_id)

            between_event_ids = filtered_between_event_ids

            self._event_graph.set_node_attributes(
                iteration_event["id"], {"events_in_iteration": list(between_event_ids)},
            )

            between_events = [
                self._event_graph.node(between_event_id)
                for between_event_id in between_event_ids
            ]

            has_parameterized_events = False
            referenced_parameters = set()
            defined_parameters = set()
            for event_data in between_events:
                _dlogger.debug("Between:  %s", node_info(event_data))
                if event_data["event_type"] == "PARAMETER_SET":
                    _dlogger.debug(
                        "There is a PARAMETER_SET event in between: %s", event_data
                    )
                    if event_data["section_name"] == section_name:
                        _dlogger.debug("And it is in this section %s", event_data)
                        has_parameterized_events = True
                    else:
                        _dlogger.debug("BUT it is not in this section %s", event_data)
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
            _dlogger.debug(
                "Section %s has parameter_list %s and refers to parameters %s ,"
                + " missing parameters %s and defines parameters %s, children defined parameters %s",
                section_name,
                parameter_list,
                referenced_parameters,
                missing_parameters,
                defined_parameters,
                defined_parameters_of_children,
            )

            defined_parameters_of_children = defined_parameters_of_children.union(
                set(parameter_list)
            )

            if (
                parameter_list == []
                and not has_parameterized_events
                and not len(missing_parameters) > 0
                and repeat_sections[section_name].num_repeats > 1
            ):
                _logger.debug("Compressing events of section %s", section_name)
                repeat_section = repeat_sections[section_name]
                self.add_iteration_control_events(
                    repeat_section, first_only=True,
                )
                _dlogger.debug(
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
                _dlogger.debug(
                    "Found %d step start events for section %s",
                    len(step_start_events),
                    section_name,
                )

                self._event_graph.set_node_attributes(
                    iteration_event["id"], {"compressed": False}
                )
                compressed = False

            for (iteration, (_, step_body_id, step_end_id),) in self._loop_step_events[
                section_name
            ].items():

                if iteration == 0:
                    for event in between_events:
                        self._event_graph.after_or_at(event["id"], step_body_id)
                elif not compressed:
                    _dlogger.debug(
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
            _dlogger.debug("Copying event %s to %s", event, new_id)

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
        spectroscopy_end_id=None,
        is_spectroscopy_signal=False,
    ):
        section_span = self._event_graph.find_section_start_end(parent_section_name)
        if section_start_node is None:
            section_start_node = section_span.start

        chain = []
        signal = None
        for i, play_wave in enumerate(play_wave_list):

            chain_element_id = parent_section_name + play_wave.id + str(i)
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
            if play_wave.signal_offset is not None:
                default_attributes["signal_offset"] = play_wave.signal_offset
            if play_wave.offset is not None:
                delay_element = ChainElement(
                    chain_element_id + "DELAY",
                    start_type=EventType.DELAY_START,
                    end_type=EventType.DELAY_END,
                    length=play_wave.offset,
                    attributes={
                        **default_attributes,
                        **{"play_wave_id": play_wave.id},
                    },
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
                attributes={**default_attributes, **{"play_wave_id": play_wave.id,},},
                start_attributes={},
            )

            if play_wave.length is not None or play_wave.is_integration:

                chain_element.start_type = EventType.PLAY_START
                chain_element.end_type = EventType.PLAY_END
                if play_wave.is_integration:
                    chain_element.start_type = EventType.ACQUIRE_START
                    chain_element.end_type = EventType.ACQUIRE_END
                if play_wave.is_delay:
                    chain_element.start_type = EventType.DELAY_START
                    chain_element.end_type = EventType.DELAY_END

                chain_element.length = play_wave.length
                chain_element.start_attributes[
                    "signal_offset"
                ] = play_wave.signal_offset
                chain_element.start_attributes["parameterized_with"] = [
                    p.param_name for p in play_wave.parameterized_with
                ]
                chain_element.start_attributes["phase"] = play_wave.phase
                chain_element.start_attributes["amplitude"] = play_wave.amplitude
                chain_element.start_attributes[
                    "acquire_handle"
                ] = play_wave.acquire_handle
                chain_element.start_attributes[
                    "acquisition_type"
                ] = play_wave.acquisition_type
                if not (play_wave.is_delay and play_wave.length is None):
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
            if not is_spectroscopy_signal and spectroscopy_end_id is not None:
                terminal_id = spectroscopy_end_id
            if is_spectroscopy_signal and spectroscopy_end_id is not None:
                pull_out_node_id = spectroscopy_end_id
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
                    _dlogger.debug(
                        "%s %s %s %s", time_beautified, event_type, signal, play_wave_id
                    )
            else:
                event_time = event["time"]
                event_type = event["event_type"]
                time_beautified = EngNumber(float(event_time))
                _dlogger.debug("%s %s", time_beautified, event_type)
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
        _dlogger.debug(
            "%sExpanding LOOP_ITERATION_END %s %s, at time %s, offset=%s",
            logheader,
            section_name,
            loop_iteration_event["id"],
            EngNumber(float(loop_iteration_event["time"])),
            EngNumber(float(offset)),
        )
        iteration_length = loop_iteration_lengths[section_name]
        iteration_start_time = loop_iteration_event["time"] - iteration_length

        _dlogger.debug(
            "%siteration_length=%s iteration_start_time=%s",
            logheader,
            EngNumber(float(iteration_length)),
            EngNumber(float(iteration_start_time)),
        )
        inner_logheader = logheader + " "
        expanded = 0
        for iteration in range(1, loop_iteration_event["num_repeats"]):

            _dlogger.debug(
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

                _dlogger.debug(
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
                    _dlogger.debug(
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
                    _dlogger.debug(
                        "%sDescending a level for iteration %d of iteration event %s  section %s",
                        inner_logheader,
                        iteration,
                        loop_iteration_event["id"],
                        section_name,
                    )
                    Compiler._expand_loop_iterations(
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
                        _dlogger.debug(
                            "%sTruncating events at %d", inner_logheader, max_events
                        )
                        break
            if max_events is not None and events_added_ref[0] >= max_events:
                _dlogger.debug("%sTruncating events at %d", inner_logheader, max_events)
                break

        _dlogger.debug(
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
        ) = Compiler._calc_loop_iteration_info(self._events_ordered())
        _dlogger.debug("Iteration lengths: %s", loop_iteration_lengths)
        too_short = set()
        for section, event in loop_iteration_ends.items():
            start_time = loop_starts[section]
            section_length = event["time"] - start_time
            section_devices = set()
            for signal in self._experiment_dao.section_signals_with_children(section):
                section_devices.add(
                    DeviceType(self._experiment_dao.signal_info(signal)["device_type"])
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
            _dlogger.debug(
                "Sections %s are too short, will expand even though expand_loops is false",
                list(too_short),
            )

        event_objects: List[Any] = []

        for event in self._events_ordered():
            _dlogger.debug("event: %s", event)
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
                    _dlogger.debug("Truncating events at %d", max_events)
                    break
                Compiler._expand_loop_iterations(
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
            EventType.PARAMETER_SET: -15,
            EventType.INCREMENT_OSCILLATOR_PHASE: -9,
        }

        for local_event_id in self._event_timing.keys():
            event = self._event_graph.node(local_event_id)

            if event["event_type"] in priority_map:
                key = (event["time"], priority_map[event["event_type"]], event["id"])
                sorted_events[key] = event

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
                    phase_incr = parameter_values[phase_incr.param_name]
                oscillator_phase_cumulative[signal_id] += phase_incr
            if event["event_type"] == EventType.SET_OSCILLATOR_PHASE:
                signal_id = event["signal"]
                osc_phase = event["set_oscillator_phase"]
                if isinstance(osc_phase, ParamRef):
                    osc_phase = parameter_values[osc_phase.param_name]
                oscillator_phase_cumulative[signal_id] = osc_phase
                oscillator_phase_sets[signal_id] = event["time"]

            if event["event_type"] == EventType.PLAY_START:
                amplitude = event["amplitude"]
                if isinstance(amplitude, ParamRef):
                    _dlogger.debug(
                        "Resolving param name %s, parameter_values=%s",
                        amplitude.param_name,
                        parameter_values,
                    )
                    amplitude = parameter_values[amplitude.param_name]
                    amplitude = (
                        round(amplitude * amplitude_resolution) / amplitude_resolution
                    )
                    if abs(amplitude) > 1.0:
                        raise LabOneQException(
                            f"Magnitude of amplitude {amplitude} cannot be larger than 1 for event {event}"
                        )

                    self._event_graph.set_node_attributes(
                        event["id"], {"amplitude": amplitude}
                    )

                phase = event["phase"]
                if isinstance(phase, ParamRef):
                    phase = parameter_values[phase.param_name]
                    self._event_graph.set_node_attributes(event["id"], {"phase": phase})

                oscillator_phase = None
                baseband_phase = None
                signal_id = event["signal"]
                signal_info = self._experiment_dao.signal_info(signal_id)
                oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
                if oscillator_info is not None:
                    if signal_info["modulation"] and signal_info["device_type"] in [
                        "hdawg",
                        "shfsg",
                    ]:
                        incremented_phase = 0.0
                        if signal_id in oscillator_phase_cumulative:
                            incremented_phase = oscillator_phase_cumulative[signal_id]

                        if oscillator_info["hardware"]:
                            if len(oscillator_phase_sets) > 0:
                                raise Exception(
                                    f"There are set_oscillator_phase entries for signal '{signal_id}', but oscillator '{oscillator_info['id']}' is a hardware oscillator. Setting absolute phase is not supported for hardware oscillators."
                                )
                            baseband_phase = incremented_phase
                        else:
                            phase_reference_time = phase_reset_time
                            if signal_id in oscillator_phase_sets:
                                phase_reference_time = max(
                                    phase_reset_time, oscillator_phase_sets[signal_id]
                                )
                            oscillator_phase = (
                                (event["time"] - phase_reference_time)
                                * 2.0
                                * math.pi
                                * oscillator_info["frequency"]
                                + incremented_phase
                            )
                event_node = self._event_graph.node(event["id"])
                event_node["oscillator_phase"] = oscillator_phase
                event_node["baseband_phase"] = baseband_phase

    def calculate_timing(self):

        (
            event_times,
            sorted_events,
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
                    loop_step_start_events = self._event_graph.find_section_events_by_type(
                        repetition_mode_auto_section,
                        event_type=EventType.LOOP_STEP_START,
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

                    _dlogger.debug(
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
                    sorted_events,
                    event_times_tiny_samples,
                ) = self._calculate_timing_for_graph(self._event_graph)

        for event_id, time in event_times.items():
            self._event_graph.node(event_id)["time"] = time

        self._event_timing = ValueSortedDict()
        sequence_nr = 0
        for event_id in sorted_events:
            event_data = self._event_graph.node(event_id)
            event_data["sequence_nr"] = sequence_nr
            event_time = float(event_data["time"])
            event_data["time"] = event_time
            self._event_timing[event_id] = (event_time, sequence_nr)
            sequence_nr += 1

        _logger.debug("Event times calculated")

    def _calculate_timing_for_graph(self, event_graph):
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

        sorted_events = event_graph.sorted_events()

        iteration_ends = {}

        for event_id in sorted_events:
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
                    raise Exception(
                        f"Section end of section {event_data['section_name'] } must happen exactly {exactly_constraint['delta']} s after its start, but it contains operations which take longer."
                    )
                if (
                    event_type in [EventType.LOOP_STEP_END]
                ) and exactly_constraint is not None:
                    iteration = event_data.get("iteration")
                    raise Exception(
                        f"Sweep step end of iteration {iteration} in section {event_data['section_name'] }  must happen exactly {exactly_constraint['delta']} s after its start, but it contains operations which take longer."
                    )

                raise Exception(
                    f"Inconsistent constraints for event {node_info(node1)}"
                )
            else:
                if corrected__latest_event_time is not None:
                    event_time = corrected__latest_event_time
                else:
                    event_time = corrected__earliest_event_time

            _dlogger.debug(
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

        return event_times, sorted_events, event_times_tiny_samples

    def verify_timing(self,):
        TOLERANCE = 1e-11
        sorted_events = sorted(
            [
                self._event_graph.node(event_id)
                for event_id in self._event_graph.node_ids()
            ],
            # PARAMETER_SET must be visited early in case of tie
            key=lambda node: (node["time"], node["event_type"] != "PARAMETER_SET"),
        )
        parameter_values = {}
        for event in sorted_events:
            if event["event_type"] == EventType.PARAMETER_SET:
                parameter_values[event["parameter"]["id"]] = event["value"]

            for _, other_node_id, edge_data in self._event_graph.out_edges(event["id"]):
                other_event = self._event_graph.node(other_node_id)
                relation = edge_data["relation"]
                if "delta" in edge_data:
                    if isinstance(edge_data["delta"], ParamRef):
                        delta = parameter_values[edge_data["delta"].param_name]
                    else:
                        delta = edge_data["delta"]
                else:
                    delta = None

                if relation == EventRelation.AFTER_OR_AT:
                    assert event["time"] >= other_event["time"]
                elif relation == EventRelation.AFTER_AT_LEAST:
                    assert event["time"] >= other_event["time"] + delta - TOLERANCE
                elif relation == EventRelation.AFTER:
                    assert event["time"] > other_event["time"]
                elif relation in (
                    EventRelation.USES_EARLY_REFERENCE,
                    EventRelation.USES_LATE_REFERENCE,
                ):
                    pass
                elif relation == EventRelation.AFTER_EXACTLY:
                    assert event["time"] - other_event["time"] - delta >= -TOLERANCE
                elif relation == EventRelation.AFTER_LOOP:
                    pass
                    # assert event["time"] == other_event["time"]
                elif relation == EventRelation.BEFORE:
                    assert event["time"] < other_event["time"]
                elif relation == EventRelation.BEFORE_OR_AT:
                    assert event["time"] <= other_event["time"]
                elif relation == EventRelation.BEFORE_AT_LEAST:
                    assert event["time"] + delta <= other_event["time"] + TOLERANCE
                elif relation == EventRelation.RELATIVE_BEFORE:
                    assert event["time"] <= other_event["time"]
                else:
                    raise ValueError("invalid relation")

    def section_grid(self, section):

        if section in self._section_grids:
            _dlogger.debug("Section cache hit for section %s", section)
            return self._section_grids[section]

        if self._settings.FIXED_SLOW_SECTION_GRID:
            section_grid = Fraction(8, int(600e6))
            _logger.warning(
                "Fixed section grid because setting 'fixed_slow_section_grid' is set"
            )
            self._section_grids[section] = section_grid
            return section_grid

        if self._section_graph_object is not None:
            section_display_name = self._section_graph_object.section_info(section)[
                "section_display_name"
            ]
        else:
            section_display_name = section

        device_types = [
            DeviceType(device_type)
            for device_type in self._experiment_dao.device_types_in_section(
                section_display_name
            )
        ]
        section_info = self._experiment_dao.section_info(section_display_name)
        has_control_elements = section_info["has_repeat"]
        has_control_elements = has_control_elements or (
            section_info["trigger"] is not None and len(section_info["trigger"]) > 0
        )

        _dlogger.debug("Device types for section %s are %s", section, device_types)
        self._section_grids[section] = Compiler._calculate_section_grid(
            section,
            device_types,
            self._clock_settings["use_2GHz_for_HDAWG"],
            has_control_elements,
        )
        return self._section_grids[section]

    @staticmethod
    def _calculate_section_grid(
        section, device_types, use_2GHz_for_HDAWG, has_control_elements
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
        if len(signal_frequencies) == 1 and not has_control_elements:
            _dlogger.debug(
                "signal frequencies are %s - returning signal frequency grid",
                [str(EngNumber(f)) for f in signal_frequencies],
            )
            return 1 / signal_frequencies[0]

        _dlogger.debug(
            "Sequencer frequencies are %s",
            [str(EngNumber(f)) for f in sequencer_frequencies],
        )
        sequencer_frequency = np.gcd.reduce(sequencer_frequencies)
        return 1 / Fraction(int(sequencer_frequency))

    def print_section_graph(self):
        self._section_graph_object.log_graph()

    def use_experiment(self, experiment):
        if "experiment" in experiment and "setup" in experiment:
            _logger.debug("Processing DSLv3 setup and experiment")
            self._experiment_dao = ExperimentDAO(
                None, experiment["setup"], experiment["experiment"]
            )
        else:
            self._experiment_dao = ExperimentDAO(experiment)

    def _analyze_setup(self):
        def get_first_instr_of(device_infos, type):
            return next(
                (instr for instr in device_infos if instr["device_type"] == type)
            )

        device_infos = self._experiment_dao.device_infos()
        device_type_list = [i["device_type"] for i in device_infos]
        type_counter = Counter(device_type_list)
        has_pqsc = type_counter["pqsc"] > 0
        has_hdawg = type_counter["hdawg"] > 0
        has_shfsg = type_counter["shfsg"] > 0
        has_shfqa = type_counter["shfqa"] > 0
        shf_types = {"shfsg", "shfqa", "shfqc"}
        has_shf = bool(shf_types.intersection(set(device_type_list)))

        # Basic validity checks
        signal_infos = [
            self._experiment_dao.signal_info(signal_id)
            for signal_id in self._experiment_dao.signals()
        ]
        used_devices = set(info["device_type"] for info in signal_infos)
        used_device_serials = set(info["device_serial"] for info in signal_infos)
        if (
            "hdawg" in used_devices
            and "uhfqa" in used_devices
            and bool(shf_types.intersection(used_devices))
        ):
            raise RuntimeError(
                "Setups with signals on each of HDAWG, UHFQA and SHF type "
                + "instruments are not supported"
            )

        self._leader_properties.is_desktop_setup = not has_pqsc and (
            used_devices == {"hdawg"}
            or used_devices == {"shfsg"}
            or used_devices == {"shfqa"}
            or used_devices == {"shfqa", "shfsg"}
            and len(used_device_serials) == 1  # SHFQC
            or used_devices == {"hdawg", "uhfqa"}
            or (used_devices == {"uhfqa"} and has_hdawg)  # No signal on leader
            or (used_devices == {"shfsg"})
        )
        if (
            not has_pqsc
            and not self._leader_properties.is_desktop_setup
            and used_devices != {"uhfqa"}
            and bool(used_devices)  # Allow empty experiment (used in tests)
        ):
            raise RuntimeError(
                f"Unsupported device combination {used_devices} for small setup"
            )

        leader = self._experiment_dao.global_leader_device()
        if self._leader_properties.is_desktop_setup:
            if leader is None:
                if has_hdawg:
                    leader = get_first_instr_of(device_infos, "hdawg")["id"]
                elif has_shfqa:
                    leader = get_first_instr_of(device_infos, "shfqa")["id"]
                    if has_shfsg:  # SHFQC
                        self._leader_properties.internal_followers = [
                            get_first_instr_of(device_infos, "shfsg")["id"]
                        ]
                elif has_shfsg:
                    leader = get_first_instr_of(device_infos, "shfsg")["id"]

            _logger.debug(
                "Using desktop setup configuration with leader %s", leader,
            )

            has_signal_on_awg_0_of_leader = False
            for signal_id in self._experiment_dao.signals():
                signal_info = self._experiment_dao.signal_info(signal_id)
                if signal_info["device_id"] == leader and (
                    0 in signal_info["channels"] or 1 in signal_info["channels"]
                ):
                    has_signal_on_awg_0_of_leader = True
                    break

            if not has_signal_on_awg_0_of_leader:
                signal_id = "__small_system_trigger__"
                device_id = leader
                signal_type = "iq"
                channels = [0, 1]
                self._experiment_dao.add_signal(
                    device_id, channels, "out", signal_id, signal_type, False
                )
                _logger.debug(
                    "No pulses played on channels 1 or 2 of %s, adding dummy signal %s to ensure triggering of the setup",
                    leader,
                    signal_id,
                )

            has_qa = type_counter["shfqa"] > 0 or type_counter["uhfqa"] > 0
            is_hdawg_solo = type_counter["hdawg"] == 1 and not has_shf and not has_qa
            if is_hdawg_solo:
                first_hdawg = get_first_instr_of(device_infos, "hdawg")
                if first_hdawg["reference_clock_source"] is None:
                    self._clock_settings[first_hdawg["id"]] = "internal"
            else:
                if not has_hdawg and has_shfsg:  # SHFSG or SHFQC solo
                    first_shfsg = get_first_instr_of(device_infos, "shfsg")
                    if first_shfsg["reference_clock_source"] is None:
                        self._clock_settings[first_shfsg["id"]] = "internal"
                if not has_hdawg and has_shfqa:  # SHFQA or SHFQC solo
                    first_shfqa = get_first_instr_of(device_infos, "shfqa")
                    if first_shfqa["reference_clock_source"] is None:
                        self._clock_settings[first_shfqa["id"]] = "internal"

        self._clock_settings["use_2GHz_for_HDAWG"] = has_shf
        self._leader_properties.global_leader = leader

    def _process_experiment(self, experiment):
        self._sampling_rate_cache = {}
        self.use_experiment(experiment)
        self._analyze_setup()
        self._calc_osc_numbering()
        self._calc_awgs()
        _dlogger.debug("Processing Sections:::::::")
        self._section_graph_object = SectionGraph.from_dao(self._experiment_dao)
        self.print_section_graph()

        root_sections = self._section_graph_object.root_sections()

        start_events = self._add_start_events()

        EventGraphBuilder.build_section_structure(
            self._event_graph, self._section_graph_object, start_events
        )

        for section_node in self._section_graph_object.topologically_sorted_sections():

            parent = self._section_graph_object.parent(section_node)
            assert parent is not None or section_node in root_sections
            section_info = self._section_graph_object.section_info(section_node)
            section_name = section_info["section_id"]

            spectroscopy_trigger_signals = set()
            if section_info["trigger"] is not None:
                if "spectroscopy" in section_info["trigger"]:
                    for device in self._experiment_dao.devices_in_section_no_descend(
                        section_info["section_display_name"]
                    ):
                        for signal_name in self._experiment_dao.device_signals(device):
                            if (
                                self._experiment_dao.signal_info(signal_name)[
                                    "signal_type"
                                ]
                                == "integration"
                            ):
                                spectroscopy_trigger_signals.add(signal_name)

            _dlogger.debug(
                "Calculating signal offsets for section %s, spectroscopy signals are: %s",
                section_name,
                spectroscopy_trigger_signals,
            )

            section_span = self._event_graph.find_section_start_end(section_name)

            section_start_node = section_span.start

            spectroscopy_end_id = None
            is_spectroscopy_section = False
            if len(spectroscopy_trigger_signals) > 0:
                is_spectroscopy_section = True
                spectroscopy_end_id = self._event_graph.find_section_events_by_type(
                    section_name, EventType.SPECTROSCOPY_END
                )[0]["id"]

            signals = self._experiment_dao.section_signals(
                section_info["section_display_name"]
            )

            signals = signals.union(spectroscopy_trigger_signals)
            for signal_id in signals:
                _dlogger.debug(
                    "Considering signal %s in section %s", signal_id, section_name,
                )

                signal_offset = None
                signal_info_main = self._experiment_dao.signal_info(signal_id)

                _dlogger.debug("signal_info_main =  %s", signal_info_main)
                play_wave_chain: List[_PlayWave] = []

                is_integration = signal_info_main["signal_type"] == "integration"

                suppress_offset_indices = []
                if is_integration:
                    if section_info["align"] == "right":
                        if (
                            len(
                                self._experiment_dao.section_pulses(
                                    section_info["section_display_name"], signal_id
                                )
                            )
                            > 0
                            and DeviceType(signal_info_main["device_type"])
                            == DeviceType.SHFQA
                        ):
                            raise RuntimeError(
                                f"Right-aligned section {section_info['section_display_name']} found containing SHFQA integration signal {signal_id} - SHFQA only supports left-aligned sections for measurement"
                            )
                    else:
                        integration_offset = 0.0

                        for pulse_index, section_pulse in enumerate(
                            self._experiment_dao.section_pulses(
                                section_info["section_display_name"], signal_id
                            )
                        ):
                            if section_pulse["offset"] is not None:
                                integration_offset += section_pulse["offset"]
                                suppress_offset_indices.append(pulse_index)
                        signal_offset = integration_offset

                if signal_offset is not None:
                    play_wave = _PlayWave(
                        id="offset_" + signal_id,
                        signal=signal_id,
                        offset=signal_offset,
                        is_delay=True,
                    )
                    play_wave_chain.append(play_wave)

                section_offset = section_info.get("offset")
                section_offset_param = section_info.get("offset_param")

                if section_offset is not None or section_offset_param is not None:
                    _dlogger.debug(
                        "Section offset for section %s : %s, param: %s",
                        section_name,
                        section_offset,
                        section_offset_param,
                    )

                    play_wave = _PlayWave(
                        id=f"section_offset_{section_name}_{signal_id}",
                        signal=signal_id,
                        acquisition_type=["spectroscopy"],
                    )

                    if section_offset is not None:
                        play_wave.offset = section_offset
                    elif section_offset_param is not None:
                        param_ref = ParamRef(section_offset_param)
                        play_wave.offset = param_ref
                    play_wave_chain.append(play_wave)

                if signal_id in spectroscopy_trigger_signals:
                    play_wave = _PlayWave(
                        id="spectroscopy_" + signal_id,
                        signal=signal_id,
                        signal_offset=signal_offset,
                        acquisition_type=["spectroscopy"],
                    )
                    play_wave.is_integration = True
                    play_wave_chain.append(play_wave)
                    _dlogger.debug("Added play_wave=%s to play wave chain", play_wave)

                for pulse_index, section_pulse in enumerate(
                    self._experiment_dao.section_pulses(
                        section_info["section_display_name"], signal_id
                    )
                ):
                    pulse_name = section_pulse["pulse_id"]
                    pulse_def = None
                    if pulse_name is not None:
                        pulse_def = self._experiment_dao.pulse(pulse_name)
                        play_wave = _PlayWave(id=pulse_name, signal=signal_id)
                    else:
                        pulse_name = "DELAY"
                        play_wave = _PlayWave(
                            id=pulse_name, signal=signal_id, is_delay=True
                        )

                    assert signal_id == signal_info_main["signal_id"]

                    device_id = signal_info_main["device_id"]

                    sampling_rate = self._sampling_rate_for_device(device_id)
                    length = Compiler.get_length_from_pulse_def(
                        pulse_def, sampling_rate
                    )
                    play_wave.is_integration = is_integration
                    if is_integration and signal_offset is not None:
                        play_wave.signal_offset = signal_offset

                    play_wave.parameterized_with = []
                    _set_playwave_value_or_amplitude(section_pulse, play_wave, "offset")
                    if (
                        section_pulse["offset"] is not None
                        and pulse_index in suppress_offset_indices
                    ):
                        play_wave.offset = 0.0
                    _set_playwave_value_or_amplitude(
                        section_pulse, play_wave, "amplitude"
                    )
                    _set_playwave_value_or_amplitude(
                        section_pulse, play_wave, "length", length
                    )
                    _set_playwave_value_or_amplitude(section_pulse, play_wave, "phase")
                    _set_playwave_value_or_amplitude(
                        section_pulse, play_wave, "increment_oscillator_phase"
                    )
                    _set_playwave_value_or_amplitude(
                        section_pulse, play_wave, "set_oscillator_phase"
                    )

                    if section_pulse["acquire_params"] is not None:
                        play_wave.acquire_handle = section_pulse["acquire_params"][
                            "handle"
                        ]
                        play_wave.acquisition_type.append(
                            section_pulse["acquire_params"]["acquisition_type"]
                        )

                    play_wave_chain.append(play_wave)

                self.add_play_wave_chain(
                    play_wave_chain,
                    section_info["section_id"],
                    right_aligned=(section_info["align"] == "right"),
                    section_start_node=section_start_node,
                    is_spectroscopy_signal=is_integration and is_spectroscopy_section,
                    spectroscopy_end_id=spectroscopy_end_id,
                )

        repeat_sections: Dict[str, Compiler.RepeatSectionsEntry] = {}
        for section_node in list(
            reversed(list(self._section_graph_object.topologically_sorted_sections()))
        ):
            _dlogger.debug("Processing section %s", section_node)
            section = self._section_graph_object.section_info(section_node)

            if section["has_repeat"]:
                section_name = section["section_id"]
                _dlogger.debug("Adding repeat for section %s", section_name)
                parameters_list = [
                    {
                        "id": param["id"],
                        "start": param["start"],
                        "step": param["step"],
                        "values": param["values"],
                    }
                    for param in self._experiment_dao.section_parameters(
                        section["section_display_name"]
                    )
                ]
                num_repeats = section["count"]

                reset_phase_hw = section["reset_oscillator_phase"]
                reset_phase_sw = (
                    reset_phase_hw or section["averaging_type"] == "hardware"
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
        self.calculate_timing()
        self.verify_timing()
        _logger.debug("Calculating play wave parameters")
        self._calculate_play_wave_parameters()

    def _find_repetition_mode_info(self, section):
        section_info = self._section_graph_object.section_info(section)
        if section_info.get("averaging_mode") == "sequential" and (
            section_info.get("repetition_mode") == "constant"
            or section_info.get("repetition_mode") == "auto"
        ):
            return {
                k: section_info.get(k)
                for k in [
                    "repetition_mode",
                    "repetition_time",
                    "averaging_mode",
                    "section_id",
                ]
            }

        has_repeating_child = False
        for child in self._section_graph_object.section_children(section):
            child_section_info = self._section_graph_object.section_info(child)
            if child_section_info.get("has_repeat"):
                has_repeating_child = True

        if not has_repeating_child and (
            section_info.get("repetition_mode") == "constant"
            or section_info.get("repetition_mode") == "auto"
        ):
            return {
                k: section_info.get(k)
                for k in [
                    "repetition_mode",
                    "repetition_time",
                    "averaging_mode",
                    "section_id",
                ]
            }

        if not section_info.get("has_repeat"):
            return None

        last_parent_section = None
        while True:
            parent_section = self._section_graph_object.parent(section)
            if parent_section is None or parent_section == last_parent_section:
                return None
            section_info = self._section_graph_object.section_info(parent_section)
            if (
                section_info.get("repetition_mode") == "constant"
                or section_info.get("repetition_mode") == "auto"
            ) and section_info.get("averaging_mode") == "cyclic":

                return {
                    k: section_info.get(k)
                    for k in [
                        "repetition_mode",
                        "repetition_time",
                        "averaging_mode",
                        "section_id",
                    ]
                }

            last_parent_section = parent_section

    def _add_start_events(self):
        retval = []

        # Add initial events to reset the NCOs.
        # Todo (PW): Drop once system tests have been migrated from legacy behaviour.
        for device_info in self._experiment_dao.device_infos():
            try:
                device_type = DeviceType(device_info["device_type"])
            except ValueError:
                # Not every device has a corresponding DeviceType (e.g. PQSC)
                continue
            if not device_type.supports_reset_osc_phase:
                continue
            retval.append(
                self._event_graph.add_node(
                    event_type=EventType.INITIAL_RESET_HW_OSCILLATOR_PHASE,
                    device_id=device_info["id"],
                    duration=device_type.reset_osc_duration,
                )
            )
        return retval

    @staticmethod
    def _get_total_rounded_delay(delay, signal_id, device_type, sampling_rate):
        if delay < 0:
            raise RuntimeError(
                f"Negative signal delay for signal {signal_id} specified."
            )
        # Quantize to granularity and round ties towards zero
        samples = delay * sampling_rate
        samples_rounded = (
            math.ceil(samples / device_type.sample_multiple + 0.5) - 1
        ) * device_type.sample_multiple
        delay_rounded = samples_rounded / sampling_rate
        if abs(samples - samples_rounded) > 1:
            _logger.debug(
                "Signal delay %.2f ns of %s on a %s will be rounded to "
                + "%.2f ns, a multiple of %d samples.",
                delay * 1e9,
                signal_id,
                device_type.name,
                delay_rounded * 1e9,
                device_type.sample_multiple,
            )
        return delay_rounded

    def _sampling_rate_for_device(self, device_id):
        if device_id not in self._sampling_rate_cache:

            device_type = DeviceType(
                self._experiment_dao.device_info(device_id)["device_type"]
            )
            if (
                device_type == DeviceType.HDAWG
                and self._clock_settings["use_2GHz_for_HDAWG"]
            ):
                sampling_rate = DeviceType.HDAWG.sampling_rate_2GHz
            else:
                sampling_rate = device_type.sampling_rate
            self._sampling_rate_cache[device_id] = sampling_rate
        else:
            sampling_rate = self._sampling_rate_cache[device_id]

        return sampling_rate

    def _generate_code(self, signal_delays):
        self._calc_awgs()
        pulse_defs = self._experiment_dao.pulses_dict()

        pulse_def_dict = {}
        for k, v in pulse_defs.items():
            pulse_obj = {}
            pulse_obj["id"] = k
            pulse_obj["function"] = v["function"]
            pulse_obj["length"] = v["length"]
            amplitude = 1.0
            if "amplitude" in v and v["amplitude"] is not None:
                amplitude = v["amplitude"]
            pulse_obj["amplitude"] = amplitude
            pulse_obj["samples"] = v.get("samples")

            pulse_def_dict[k] = pulse_obj

        self._calc_shfqa_generator_allocation()

        code_generator = CodeGenerator(self._settings)
        self._code_generator = code_generator

        for signal_id in self._experiment_dao.signals():

            signal_info = self._experiment_dao.signal_info(signal_id)
            delay_signal = signal_info["delay_signal"]

            device_type = DeviceType(signal_info["device_type"])
            device_id = signal_info["device_id"]

            sampling_rate = self._sampling_rate_for_device(device_id)
            delay = self.get_delay(
                device_type,
                self._leader_properties.is_desktop_setup,
                self._clock_settings["use_2GHz_for_HDAWG"],
            )
            if delay_signal is not None:
                delay_signal = self._get_total_rounded_delay(
                    delay_signal, signal_id, device_type, sampling_rate
                )
                delay += delay_signal

            if signal_id in signal_delays:
                delay += signal_delays[signal_id]["code_generation"]

            trigger_mode = TriggerMode.NONE
            device_info = self._experiment_dao.device_info(device_id)
            try:
                reference_clock_source = self._clock_settings[device_id]
            except KeyError:
                reference_clock_source = device_info["reference_clock_source"]
            if self._leader_properties.is_desktop_setup:
                trigger_mode = {
                    DeviceType.HDAWG: TriggerMode.DIO_TRIGGER,
                    DeviceType.SHFSG: TriggerMode.INTERNAL_TRIGGER_WAIT,
                    DeviceType.SHFQA: TriggerMode.INTERNAL_TRIGGER_WAIT,
                    DeviceType.UHFQA: TriggerMode.DIO_WAIT,
                }.get(device_type, TriggerMode.NONE)

            awg = self.get_awg(signal_id)
            signal_type = signal_info["signal_type"]

            _dlogger.debug(
                "Adding signal %s with signal type %s", signal_id, signal_type
            )

            oscillator_frequency = None

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            if (
                oscillator_info is not None
                and not oscillator_info["hardware"]
                and signal_info["modulation"]
            ):
                oscillator_frequency = oscillator_info["frequency"]
            channels = copy.deepcopy(signal_info["channels"])
            if signal_id in self._integration_unit_allocation:
                channels = copy.deepcopy(
                    self._integration_unit_allocation[signal_id]["channels"]
                )
            elif signal_id in self._shfqa_generator_allocation:
                channels = copy.deepcopy(
                    self._shfqa_generator_allocation[signal_id]["channels"]
                )

            signal_obj = SignalObj(
                id=signal_id,
                sampling_rate=sampling_rate,
                delay=delay,
                signal_type=signal_type,
                device_id=device_id,
                awg=awg,
                device_type=device_type,
                oscillator_frequency=oscillator_frequency,
                trigger_mode=trigger_mode,
                reference_clock_source=reference_clock_source,
                channels=channels,
            )
            code_generator.add_signal(signal_obj)

        _logger.debug("Preparing events for code generator")
        events = self.event_timing(expand_loops=False)
        code_generator.gen_acquire_map(events, self._section_graph_object)
        code_generator.gen_seq_c(events, pulse_def_dict)
        code_generator.gen_waves()

        _logger.debug("Code generation completed")

    def _calc_osc_numbering(self):
        self._osc_numbering = {}

        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_type = DeviceType(signal_info["device_type"])

            if signal_info["signal_type"] == "integration":
                continue

            hw_osc_names = set()
            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            if oscillator_info is not None and oscillator_info["hardware"]:
                hw_osc_names.add(oscillator_info["id"])

            base_channel = min(signal_info["channels"])
            min_osc_number = base_channel * 2
            count = 0
            for osc_name in hw_osc_names:
                if device_type == DeviceType.SHFQA:
                    self._osc_numbering[osc_name] = min(signal_info["channels"])
                else:
                    self._osc_numbering[osc_name] = min_osc_number + count
                    count += 1

    def _calc_integration_unit_allocation(self):
        self._integration_unit_allocation = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            _dlogger.debug("_integration_unit_allocation considering %s", signal_info)
            if signal_info["signal_type"] == "integration":
                _dlogger.debug(
                    "_integration_unit_allocation: found integration signal %s",
                    signal_info,
                )
                device_id = signal_info["device_id"]
                device_type = DeviceType(signal_info["device_type"])
                awg_nr = Compiler.calc_awg_number(
                    signal_info["channels"][0], device_type
                )

                num_acquire_signals = len(
                    list(
                        filter(
                            lambda x: x["device_id"] == device_id
                            and x["awg_nr"] == awg_nr,
                            self._integration_unit_allocation.values(),
                        )
                    )
                )

                integrators_per_signal = (
                    device_type.num_integration_units_per_acquire_signal
                    if self._experiment_dao.acquisition_type
                    in [
                        AcquisitionType.RAW,
                        AcquisitionType.SPECTROSCOPY,
                        AcquisitionType.INTEGRATION,
                    ]
                    else 1
                )

                self._integration_unit_allocation[signal_id] = {
                    "device_id": device_id,
                    "awg_nr": awg_nr,
                    "channels": [
                        integrators_per_signal * num_acquire_signals + i
                        for i in range(integrators_per_signal)
                    ],
                }

    def _calc_shfqa_generator_allocation(self):
        self._shfqa_generator_allocation = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_type = DeviceType(signal_info["device_type"])

            if signal_info["signal_type"] == "iq" and device_type == DeviceType.SHFQA:
                _dlogger.debug(
                    "_shfqa_generator_allocation: found SHFQA iq signal %s", signal_info
                )
                device_id = signal_info["device_id"]
                awg_nr = Compiler.calc_awg_number(
                    signal_info["channels"][0], device_type
                )
                num_generator_signals = len(
                    list(
                        filter(
                            lambda x: x["device_id"] == device_id
                            and x["awg_nr"] == awg_nr,
                            self._shfqa_generator_allocation.values(),
                        )
                    )
                )

                self._shfqa_generator_allocation[signal_id] = {
                    "device_id": device_id,
                    "awg_nr": awg_nr,
                    "channels": [num_generator_signals],
                }

    def osc_number(self, osc_name):
        if self._osc_numbering is None:
            raise Exception(f"Oscillator numbers not yet calculated")
        return self._osc_numbering[osc_name]

    @staticmethod
    def calc_awg_number(channel, device_type: DeviceType):
        if device_type == DeviceType.UHFQA:
            return 0
        return int(math.floor(channel / device_type.channels_per_awg))

    def _calc_awgs(self):
        awgs: _AWGMapping = {}
        signals_by_channel_and_awg: Dict[
            Tuple[str, int, int], Dict[str, Union[Set, AWGInfo]]
        ] = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_id = signal_info["device_id"]
            device_type = DeviceType(signal_info["device_type"])
            for channel in sorted(signal_info["channels"]):
                awg_number = Compiler.calc_awg_number(channel, device_type)
                device_awgs = awgs.setdefault(device_id, SortedDict())
                awg = device_awgs.get(awg_number)
                if awg is None:
                    signal_type = signal_info["signal_type"]
                    # Treat "integration" signal type same as "iq" at AWG level
                    if signal_type == "integration":
                        signal_type = "iq"
                    awg = AWGInfo(
                        device_id=device_id,
                        signal_type=AWGSignalType(signal_type),
                        awg_number=awg_number,
                        seqc="seq_" + device_id + "_" + str(awg_number) + ".seqc",
                        device_type=device_type,
                    )
                    device_awgs[awg_number] = awg

                awg.signal_channels.append((signal_id, channel))

                if signal_info["signal_type"] == "iq":
                    signal_channel_awg_key = (device_id, awg.awg_number, channel)
                    if signal_channel_awg_key in signals_by_channel_and_awg:
                        signals_by_channel_and_awg[signal_channel_awg_key][
                            "signals"
                        ].add(signal_id)
                    else:
                        signals_by_channel_and_awg[signal_channel_awg_key] = {
                            "awg": awg,
                            "signals": {signal_id},
                        }

        for v in signals_by_channel_and_awg.values():
            if len(v["signals"]) > 1 and v["awg"].device_type != DeviceType.SHFQA:
                awg = v["awg"]
                awg.signal_type = AWGSignalType.MULTI
                _dlogger.debug(f"Changing signal type to multi: {awg}")

        for dev_awgs in awgs.values():
            for awg in dev_awgs.values():
                _dlogger.debug(f"Consider awg: {awg}")
                if len(awg.signal_channels) > 1 and awg.signal_type not in [
                    AWGSignalType.IQ,
                    AWGSignalType.MULTI,
                ]:
                    awg.signal_type = AWGSignalType.DOUBLE
                    _dlogger.debug(f"Changing signal type to double: {awg}")

                # For each awg of a HDAWG, retrieve the delay of all of its rf_signals (for
                # playZeros and check whether they are the same:
                if awg.signal_type == AWGSignalType.IQ:
                    continue
                signal_ids = set(sc[0] for sc in awg.signal_channels)
                signal_delays = {
                    self._experiment_dao.signal_info(signal_id)["delay_signal"] or 0.0
                    for signal_id in signal_ids
                }
                if len(signal_delays) > 1:
                    delay_strings = ", ".join(
                        [f"{d * 1e9:.2f} ns" for d in signal_delays]
                    )
                    raise RuntimeError(
                        "Delays {" + str(delay_strings) + "} on awg "
                        f"{awg.device_id}:{awg.awg_number} with signals "
                        f"{signal_ids} differ."
                    )

        self._awgs = awgs

    def get_awg(self, signal_id) -> AWGInfo:
        awg_number = None
        signal_info = self._experiment_dao.signal_info(signal_id)

        device_id = signal_info["device_id"]
        device_type = DeviceType(signal_info["device_type"])
        awg_number = Compiler.calc_awg_number(signal_info["channels"][0], device_type)
        if (
            signal_info["signal_type"] == "integration"
            and device_type != DeviceType.SHFQA
        ):
            awg_number = 0
        return self._awgs[device_id][awg_number]

    def calc_outputs(self, signal_delays):
        all_channels = {}

        flipper = [1, 0]

        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            if signal_info["signal_type"] == "integration":
                continue
            oscillator_frequency = None
            oscillator_number = None

            oscillator_info = self._experiment_dao.signal_oscillator(signal_id)
            oscillator_is_hardware = (
                oscillator_info is not None and oscillator_info["hardware"]
            )
            if oscillator_is_hardware:
                osc_name = oscillator_info["id"]
                oscillator_frequency = oscillator_info["frequency"]
                oscillator_number = self.osc_number(osc_name)

            mixer_calibration = self._experiment_dao.mixer_calibration(signal_id)
            lo_frequency = self._experiment_dao.lo_frequency(signal_id)
            port_mode = self._experiment_dao.port_mode(signal_id)
            signal_range = self._experiment_dao.signal_range(signal_id)
            port_delay = self._experiment_dao.port_delay(signal_id)
            if signal_id in signal_delays:
                if port_delay is not None:
                    port_delay += signal_delays[signal_id]["on_device"]
                else:
                    port_delay = signal_delays[signal_id]["on_device"]

            base_channel = min(signal_info["channels"])
            for channel in signal_info["channels"]:
                output = {
                    "device_id": signal_info["device_id"],
                    "channel": channel,
                    "lo_frequency": lo_frequency,
                    "port_mode": port_mode,
                    "range": signal_range,
                    "port_delay": port_delay,
                }
                signal_is_modulated = signal_info["modulation"]
                output_modulation_logic = {
                    (True, True): True,
                    (False, False): False,
                    (True, False): False,
                    (False, True): False,
                }
                if output_modulation_logic[
                    (oscillator_is_hardware, signal_is_modulated)
                ]:
                    output["modulation"] = True
                    if oscillator_frequency is None:
                        oscillator_frequency = 0
                else:
                    output["modulation"] = False

                # default mixer calib
                if (
                    DeviceType(signal_info["device_type"]) == DeviceType.HDAWG
                ):  # for hdawgs, we add default values to the recipe
                    output["offset"] = 0.0
                    output["diagonal"] = 1.0
                    output["off_diagonal"] = 0.0
                else:  # others get no mixer calib values
                    output["offset"] = 0.0
                    output["diagonal"] = None
                    output["off_diagonal"] = None

                if signal_info["signal_type"] == "iq" and mixer_calibration is not None:
                    if mixer_calibration["voltage_offsets"] is not None:
                        output["offset"] = mixer_calibration["voltage_offsets"][
                            channel - base_channel
                        ]
                    if mixer_calibration["correction_matrix"] is not None:
                        output["diagonal"] = mixer_calibration["correction_matrix"][
                            channel - base_channel
                        ][channel - base_channel]
                        output["off_diagonal"] = mixer_calibration["correction_matrix"][
                            flipper[channel - base_channel]
                        ][channel - base_channel]

                output["oscillator_frequency"] = oscillator_frequency
                output["oscillator"] = oscillator_number
                channel_key = (signal_info["device_id"], channel)
                # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
                all_channels[channel_key] = output
        retval = sorted(
            all_channels.values(),
            key=lambda output: output["device_id"] + str(output["channel"]),
        )
        return retval

    def calc_inputs(self):
        all_channels = {}
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            if signal_info["signal_type"] != "integration":
                continue

            lo_frequency = self._experiment_dao.lo_frequency(signal_id)
            signal_range = self._experiment_dao.signal_range(signal_id)
            port_delay = self._experiment_dao.port_delay(signal_id)
            for channel in signal_info["channels"]:
                input = {
                    "device_id": signal_info["device_id"],
                    "channel": channel,
                    "lo_frequency": lo_frequency,
                    "range": signal_range,
                    "port_delay": port_delay,
                }
                channel_key = (signal_info["device_id"], channel)
                # TODO(2K): check for conflicts if 'channel_key' already present in 'all_channels'
                all_channels[channel_key] = input
        retval = sorted(
            all_channels.values(),
            key=lambda input: input["device_id"] + str(input["channel"]),
        )
        return retval

    def calc_measurement_map(self, integration_times):
        measurement_sections = []
        for (
            graph_section_name
        ) in self._section_graph_object.topologically_sorted_sections():
            graph_section_info = self._section_graph_object.section_info(
                graph_section_name
            )
            section_name = graph_section_info["section_display_name"]
            section_info = self._experiment_dao.section_info(section_name)
            if section_info["trigger"] is not None:
                measurement_sections.append(section_name)

        section_measurement_infos = []

        for section_name in measurement_sections:
            section_signals = self._experiment_dao.section_signals_with_children(
                section_name
            )

            def empty_device():
                return {
                    "signals": set(),
                    "monitor": None,
                }

            infos_by_device_awg = {}
            for signal in section_signals:
                signal_info_for_section = self._experiment_dao.signal_info(signal)
                device_type = DeviceType(signal_info_for_section["device_type"])
                awg_nr = Compiler.calc_awg_number(
                    signal_info_for_section["channels"][0], device_type
                )

                if signal_info_for_section["signal_type"] == "integration":
                    device_id = signal_info_for_section["device_id"]
                    device_awg_key = (device_id, awg_nr)
                    if device_awg_key not in infos_by_device_awg:
                        infos_by_device_awg[device_awg_key] = {
                            "section_name": section_name,
                            "devices": {},
                        }
                    section_measurement_info = infos_by_device_awg[device_awg_key]

                    device = section_measurement_info["devices"].setdefault(
                        (device_id, awg_nr), empty_device()
                    )
                    device["signals"].add(signal)

                    _dlogger.debug(
                        "Added measurement device %s",
                        signal_info_for_section["device_id"],
                    )

            section_measurement_infos.extend(infos_by_device_awg.values())

        _dlogger.debug("Found section_measurement_infos  %s", section_measurement_infos)
        measurements = {}

        for info in section_measurement_infos:
            section_name = info["section_name"]
            loop_params = self._loop_parameters_to_root(section_name)
            hw_average_infos = list(
                map(
                    itemgetter("averaging_type"),
                    filter(itemgetter("has_repeat"), loop_params),
                )
            )
            _dlogger.debug("Found hw_average_infos  %s", hw_average_infos)

            for device_awg_nr, v in info["devices"].items():

                device_id, awg_nr = device_awg_nr
                if (device_id, awg_nr) in measurements:
                    _dlogger.debug(
                        "Expanding existing measurement record for device %s awg %d (when looking at section %s )",
                        device_id,
                        awg_nr,
                        info["section_name"],
                    )
                    measurement = measurements[(device_id, awg_nr)]
                else:
                    measurement = {
                        "length": None,
                        "delay": 0,
                    }
                    if v.get("monitor") is not None:
                        measurement["monitor"] = v.get("monitor")

                    integration_time_info = integration_times.section_info(
                        info["section_name"]
                    )
                    if integration_time_info is not None:

                        _dlogger.debug(
                            "Found integration_time_info %s", integration_time_info
                        )

                        signal_info_for_section_and_device_awg = next(
                            i
                            for i in integration_time_info.values()
                            if i.awg == awg_nr and i.device_id == device_id
                        )
                        measurement[
                            "length"
                        ] = signal_info_for_section_and_device_awg.length_in_samples
                        measurement[
                            "delay"
                        ] = signal_info_for_section_and_device_awg.delay_in_samples
                    else:
                        del measurement["length"]
                    _dlogger.debug(
                        "Added measurement %s\n  for %s", measurement, device_awg_nr
                    )

                measurements[(device_id, awg_nr)] = measurement

        retval = {}
        for device_awg_key, v in measurements.items():
            device_id, awg_nr = device_awg_key
            if device_id not in retval:
                # make sure measurements are sorted by awg_nr
                retval[device_id] = SortedDict()
            retval[device_id][awg_nr] = v
            v["channel"] = awg_nr
        for k in list(retval.keys()):
            retval[k] = list(retval[k].values())

        return retval

    def _loop_parameters_to_root(self, section_id):
        fields = [
            "section_id",
            "execution_type",
            "averaging_type",
            "count",
            "trigger",
            "has_repeat",
        ]
        return list(
            reversed(
                [
                    dict(
                        zip(
                            fields,
                            itemgetter(*fields)(
                                self._experiment_dao.section_info(section_name)
                            ),
                        )
                    )
                    for section_name in self._path_to_root(section_id)
                ]
            )
        )

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
        _dlogger.debug("Path to root from %s : %s", section_id, path_to_root)
        return path_to_root

    def _generate_recipe(self, integration_times, signal_delays):
        recipe_generator = RecipeGenerator()
        recipe_generator.from_experiment(
            self._experiment_dao, self._leader_properties, self._clock_settings,
        )

        for output in self.calc_outputs(signal_delays):
            _dlogger.debug("Adding output %s", output)
            recipe_generator.add_output(
                output["device_id"],
                output["channel"],
                output["offset"],
                output["diagonal"],
                output["off_diagonal"],
                modulation=output["modulation"],
                oscillator=output["oscillator"],
                oscillator_frequency=output["oscillator_frequency"],
                lo_frequency=output["lo_frequency"],
                port_mode=output["port_mode"],
                output_range=output["range"],
                port_delay=output["port_delay"],
            )

        for input in self.calc_inputs():
            _dlogger.debug("Adding input %s", input)
            recipe_generator.add_input(
                input["device_id"],
                input["channel"],
                lo_frequency=input["lo_frequency"],
                input_range=input["range"],
                port_delay=input["port_delay"],
            )

        for device_id, awgs in self._awgs.items():
            for awg in awgs.values():
                signal_type = awg.signal_type
                if signal_type == AWGSignalType.DOUBLE:
                    signal_type = AWGSignalType.SINGLE
                if signal_type == AWGSignalType.MULTI:
                    signal_type = AWGSignalType.IQ
                recipe_generator.add_awg(
                    device_id, awg.awg_number, signal_type.value, awg.seqc
                )

        if self._code_generator is None:
            raise Exception("Code generator not initialized")
        recipe_generator.add_oscillator_params(self._experiment_dao)
        recipe_generator.add_integrator_allocations(
            self._integration_unit_allocation,
            self._experiment_dao,
            self._code_generator.integration_weights(),
        )

        signal_info_map = {
            s: self._experiment_dao.signal_info(s)
            for s in self._experiment_dao.signals()
        }
        for s in signal_info_map.values():
            s["awg_number"] = self.get_awg(s["signal_id"]).awg_number

        recipe_generator.add_acquire_lengths(integration_times)

        recipe_generator.add_measurements(
            self.calc_measurement_map(integration_times=integration_times,)
        )

        recipe_generator.add_simultaneous_acquires(
            self._code_generator.simultaneous_acquires()
        )

        recipe_generator.add_total_execution_time(
            self._code_generator.total_execution_time()
        )

        self._recipe = recipe_generator.recipe()
        _logger.debug("Recipe generation completed")

    def compiler_output(self) -> CompiledExperiment:
        return CompiledExperiment(
            recipe=self._recipe,
            src=self._code_generator.src(),
            waves=self._code_generator.waves(),
            wave_indices=self._code_generator.wave_indices(),
            schedule=self._prepare_schedule(),
            experiment_dict=ExperimentDAO.dump(self._experiment_dao),
            pulse_map=self._code_generator.pulse_map(),
        )

    def _prepare_schedule(self):
        event_list = self.event_timing(
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=self._settings.MAX_EVENTS_TO_PUBLISH,
        )

        event_list = [
            {k: v for k, v in event.items() if v is not None} for event in event_list
        ]

        section_graph = []
        section_info = {}
        subsection_map = {}

        root_section = self._section_graph_object.root_section()
        if root_section is None:
            return {
                "event_list": [],
                "section_graph": {},
                "section_info": {},
                "subsection_map": {},
                "section_signals_with_children": {},
                "sampling_rates": [],
            }

        section_info = {
            k: {"depth": v} for k, v in self._section_graph_object.depth_map().items()
        }

        if len(list(section_info.keys())) == 0:
            section_info = {root_section: {"depth": 0}}

        preorder_map = self._section_graph_object.preorder_map()

        section_info = {
            k: {**v, **{"preorder": preorder_map[k]}} for k, v in section_info.items()
        }

        _dlogger.debug("Section_info: %s", section_info)
        subsection_map = self._section_graph_object.subsection_map()
        _dlogger.debug("subsection_map=%s", subsection_map)
        for k, v in section_info.items():
            if len(subsection_map[k]) == 0:
                v["is_leaf"] = True
            else:
                v["is_leaf"] = False
        section_graph = self._section_graph_object.as_section_graph()
        section_signals_with_children = {}

        for section in self._section_graph_object.sections():
            section_data = self._section_graph_object.section_info(section)
            section_display_name = section_data["section_display_name"]
            section_signals_with_children[section] = list(
                self._experiment_dao.section_signals_with_children(section_display_name)
            )
            section_info[section]["section_display_name"] = section_display_name
            section_signals_with_children[section]

        sampling_rate_tuples = []
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
            device_id = signal_info["device_id"]
            device_type = signal_info["device_type"]
            sampling_rate_tuples.append(
                (device_type, int(self._sampling_rate_for_device(device_id)))
            )

        sampling_rates = [
            [list(set([d[0] for d in sampling_rate_tuples if d[1] == r])), r]
            for r in set([t[1] for t in sampling_rate_tuples])
        ]

        _logger.debug("Pulse sheet generation completed")

        return {
            "event_list": event_list,
            "section_graph": section_graph,
            "section_info": section_info,
            "subsection_map": subsection_map,
            "section_signals_with_children": section_signals_with_children,
            "sampling_rates": sampling_rates,
        }

    def write_files_from_output(self, output: CompiledExperiment):
        Path("awg").mkdir(parents=True, exist_ok=True)
        _dlogger.debug("Writing files to %s", os.path.abspath("awg"))
        recipe_file = os.path.join("awg", "recipe.json")
        _dlogger.debug("Writing %s", os.path.abspath(recipe_file))
        with open(recipe_file, "w") as f:
            f.write(json.dumps(output.recipe, indent=2))
        Path("awg/src").mkdir(parents=True, exist_ok=True)
        for src in output.src:
            filename = os.path.join("awg", "src", src["filename"])
            _dlogger.debug("Writing %s", os.path.abspath(filename))
            with open(filename, "w") as f:
                f.write(src["text"])
        for awg in output.wave_indices:
            filename = os.path.join("awg", "src", awg["filename"])
            _dlogger.debug("Writing %s", os.path.abspath(filename))
            with open(filename, "w", newline="") as f:
                iw = csv.writer(
                    f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
                )
                iw.writerow(["signature", "index", "type"])
                for signature, (index, stype) in awg["value"].items():
                    iw.writerow([signature, index, stype])

        Path("awg/waves").mkdir(parents=True, exist_ok=True)
        for wave in output.waves:
            filename = os.path.join("awg", "waves", wave["filename"])
            _dlogger.debug("Writing %s", os.path.abspath(filename))
            np.savetxt(filename, wave["samples"], delimiter=",")

    def write_files(self):
        self.write_files_from_output(self.compiler_output())
        _logger.debug(
            "Recipe, source and wave files written to %s", os.path.abspath(".")
        )

    def dump_src(self):
        for src in self.compiler_output().src:
            _logger.debug("*** %s", src["filename"])
            for line in src["text"].splitlines():
                _logger.debug(line)
        _logger.debug("END %s", src["filename"])

    def run(self, data) -> CompiledExperiment:
        _logger.debug("ES Compiler run")
        self._process_experiment(data)

        self._calc_integration_unit_allocation()
        signal_info_map = {
            s: self._experiment_dao.signal_info(s)
            for s in self._experiment_dao.signals()
        }
        for s in signal_info_map.values():
            s["awg_number"] = self.get_awg(s["signal_id"]).awg_number
            s["sampling_rate"] = self._sampling_rate_for_device(s["device_id"])

        (
            integration_times,
            signal_delays,
            delays_per_awg,
        ) = MeasurementCalculator.calculate_integration_times(
            signal_info_map, self.event_timing(expand_loops=False)
        )

        self._generate_code(signal_delays)
        self._generate_recipe(integration_times, signal_delays)

        retval = self.compiler_output()

        total_seqc_lines = 0
        for f in retval.src:
            total_seqc_lines += f["text"].count("\n")
        _logger.info("Total seqC lines generated: %d", total_seqc_lines)

        total_samples = 0
        for f in retval.waves:
            try:
                total_samples += len(f["samples"])
            except KeyError:
                pass
        _logger.info("Total sample points generated: %d", total_samples)

        _logger.info("Finished QCCS Compiler run.")
        return retval

    @staticmethod
    def get_length_from_pulse_def(pulse_def, sampling_rate):
        if pulse_def is None:
            return None
        length = pulse_def.get("length")
        if length is None:
            samples = pulse_def.get("samples")
            if samples is not None:
                length = len(samples) / sampling_rate
        return length

    def get_delay(self, device_type, desktop_setup, hdawg_uses_2GHz):
        if not isinstance(device_type, DeviceType):
            raise Exception(f"Device type {device_type} is not of type DeviceType")
        if device_type == DeviceType.HDAWG:
            if not desktop_setup:
                if hdawg_uses_2GHz:
                    return self._settings.HDAWG_LEAD_PQSC_2GHz
                else:
                    return self._settings.HDAWG_LEAD_PQSC
            else:
                if hdawg_uses_2GHz:
                    return self._settings.HDAWG_LEAD_DESKTOP_SETUP_2GHz
                else:
                    return self._settings.HDAWG_LEAD_DESKTOP_SETUP
        elif device_type == DeviceType.UHFQA:
            return self._settings.UHFQA_LEAD_PQSC
        elif device_type == DeviceType.SHFQA:
            return self._settings.SHFQA_LEAD_PQSC
        elif device_type == DeviceType.SHFSG:
            return self._settings.SHFSG_LEAD_PQSC
        else:
            raise Exception(f"Unsupported device type {device_type}")


def find_obj_by_id(object_list, id):
    for i in object_list:
        if i["id"] == id:
            return i

    return None


def node_info(node):
    play_wave = ""
    if "play_wave_id" in node:
        play_wave = node["play_wave_id"]
    section = ""
    if "section_name" in node:
        section = node["section_name"]
    skeleton_of = ""
    if "skeleton_of" in node:
        skeleton_of = " skeleton of " + node["skeleton_of"]
    node_id = "UNKNOWN_ID"
    if "id" in node:
        node_id = node["id"]
    event_type = "UNKNOWN_EVENT_TYPE"
    if "event_type" in node:
        event_type = node["event_type"]
    return (
        str(node_id) + " " + event_type + " " + section + " " + play_wave + skeleton_of
    )
