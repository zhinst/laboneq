# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Dict, Optional, TypedDict

from laboneq._observability.tracing import trace
from laboneq.compiler import CodeGenerator, CompilerSettings
from laboneq.compiler.code_generator import IntegrationTimes
from laboneq.compiler.code_generator.measurement_calculator import SignalDelays
from laboneq.compiler.code_generator.sampled_event_handler import FeedbackConnection
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.experiment_access import ExperimentDAO
from laboneq.compiler.ir.ir import IR
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.data.compilation_job import ExperimentInfo
from laboneq.data.scheduled_experiment import PulseMapEntry

_logger = logging.getLogger(__name__)


class Schedule(TypedDict):
    event_list: list[str]
    section_info: dict[str, dict]
    section_signals_with_children: dict[str, list[str]]
    sampling_rates: list[tuple[list[str], float]]

    @classmethod
    def empty(cls):
        return cls(
            event_list=[],
            section_info={},
            section_signals_with_children={},
            sampling_rates=[],
        )


@dataclass
class RealtimeCompilerOutput:
    command_table_match_offsets: Dict[AwgKey, int]
    feedback_connections: Dict[str, FeedbackConnection]
    feedback_registers: Dict[AwgKey, int]
    signal_delays: SignalDelays
    integration_weights: Dict
    integration_times: IntegrationTimes
    simultaneous_acquires: Dict[float, Dict[str, str]]
    total_execution_time: float
    src: Dict[AwgKey, Dict[str, Any]]
    waves: Dict[str, Dict[str, Any]]
    wave_indices: Dict[AwgKey, Dict[str, Any]]
    command_tables: Dict[AwgKey, Dict[str, Any]]
    pulse_map: Dict[str, PulseMapEntry]
    schedule: Dict[str, Any]


class RealtimeCompiler:
    def __init__(
        self,
        experiment_info: ExperimentInfo,
        scheduler: Scheduler,
        sampling_rate_tracker: SamplingRateTracker,
        signal_objects: Dict[str, SignalObj],
        settings: Optional[CompilerSettings] = None,
    ):
        self._experiment_info = experiment_info
        self._ir = None
        self._scheduler = scheduler
        self._sampling_rate_tracker = sampling_rate_tracker
        self._signal_objects = signal_objects
        self._settings = settings

        self._code_generator = None

    def _lower_to_ir(self):
        root = self._scheduler.generate_ir()
        return IR(
            devices=self._experiment_info.devices,
            signals=self._experiment_info.signals,
            root=root,
            global_leader_device=self._experiment_info.global_leader_device,
            pulse_defs=self._experiment_info.pulse_defs,
        )

    def _lower_ir_to_code(self, ir: IR):
        code_generator = CodeGenerator(self._settings)
        self._code_generator = code_generator

        for signal_obj in self._signal_objects.values():
            code_generator.add_signal(signal_obj)

        events = self._scheduler.generate_event_list_from_ir(
            ir=ir.root, expand_loops=False, max_events=float("inf")
        )

        code_generator.gen_acquire_map(events)
        code_generator.gen_seq_c(
            events,
            {p.uid: p for p in self._experiment_info.pulse_defs},
        )
        code_generator.gen_waves()

        _logger.debug("Code generation completed")

    @trace("compiler.generate-code()")
    def _generate_code(self):
        ir = self._lower_to_ir()
        _logger.debug("IR lowering complete")

        self._ir = ir

        if self._settings.FORCE_IR_ROUNDTRIP:
            ir.round_trip()

        self._lower_ir_to_code(ir)
        _logger.debug("lowering IR to code complete")

    def run(self, near_time_parameters: Optional[ParameterStore] = None):
        self._scheduler.run(near_time_parameters)
        self._generate_code()

        schedule = self.prepare_schedule() if self._settings.OUTPUT_EXTRAS else None

        compiler_output = RealtimeCompilerOutput(
            command_table_match_offsets=self._code_generator.command_table_match_offsets(),
            feedback_connections=self._code_generator.feedback_connections(),
            feedback_registers=self._code_generator.feedback_registers(),
            signal_delays=self._code_generator.signal_delays(),
            integration_weights=self._code_generator.integration_weights(),
            integration_times=self._code_generator.integration_times(),
            simultaneous_acquires=self._code_generator.simultaneous_acquires(),
            total_execution_time=self._code_generator.total_execution_time(),
            src=self._code_generator.src(),
            waves=self._code_generator.waves(),
            wave_indices=self._code_generator.wave_indices(),
            command_tables=self._code_generator.command_tables(),
            pulse_map=self._code_generator.pulse_map(),
            schedule=schedule,
        )

        return compiler_output

    def _lower_ir_to_schedule(self, ir: IR):
        event_list = self._scheduler.generate_event_list_from_ir(
            ir=ir.root,
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=self._settings.MAX_EVENTS_TO_PUBLISH,
        )

        event_list = [
            {k: v for k, v in event.items() if v is not None} for event in event_list
        ]

        experiment_dao = ExperimentDAO(self._experiment_info)

        try:
            root_section = experiment_dao.root_rt_sections()[0]
        except IndexError:
            return Schedule.empty()

        preorder_map = self._scheduler.preorder_map()

        section_info_out = {}

        section_signals_with_children = {}

        for section in [
            root_section,
            *experiment_dao.all_section_children(root_section),
        ]:
            section_info = experiment_dao.section_info(section)
            section_display_name = section_info.uid
            section_signals_with_children[section] = list(
                experiment_dao.section_signals_with_children(section)
            )
            section_info_out[section] = {
                "section_display_name": section_display_name,
                "preorder": preorder_map[section],
            }

        sampling_rate_tuples = []
        for signal_id in experiment_dao.signals():
            signal_info = experiment_dao.signal_info(signal_id)
            device_id = signal_info.device.uid
            device_type = signal_info.device.device_type.value
            sampling_rate_tuples.append(
                (
                    device_type,
                    int(
                        self._sampling_rate_tracker.sampling_rate_for_device(device_id)
                    ),
                )
            )

        # Group devices by sampling rate and create a backward compatible alist of those.
        sampling_rates = [
            (list({tpl[0] for tpl in grouped_tuples}), sampling_rate)
            for sampling_rate, grouped_tuples in groupby(
                sampling_rate_tuples, lambda tpl: tpl[1]
            )
        ]

        _logger.debug("Pulse sheet generation completed")

        return Schedule(
            event_list=event_list,
            section_info=section_info_out,
            section_signals_with_children=section_signals_with_children,
            sampling_rates=sampling_rates,
        )

    def prepare_schedule(self):
        if self._ir is None:
            self._ir = self._lower_to_ir()
        return self._lower_ir_to_schedule(self._ir)
