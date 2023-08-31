# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from laboneq._observability.tracing import trace
from laboneq.compiler import CodeGenerator, CompilerSettings
from laboneq.compiler.code_generator import IntegrationTimes
from laboneq.compiler.code_generator.measurement_calculator import SignalDelays
from laboneq.compiler.code_generator.sampled_event_handler import FeedbackConnection
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.experiment_access import ExperimentDAO
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.data.scheduled_experiment import PulseMapEntry

_logger = logging.getLogger(__name__)


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
        experiment_dao: ExperimentDAO,
        sampling_rate_tracker: SamplingRateTracker,
        signal_objects: Dict[str, SignalObj],
        settings: Optional[CompilerSettings] = None,
    ):
        self._experiment_dao = experiment_dao
        self._sampling_rate_tracker = sampling_rate_tracker
        self._signal_objects = signal_objects
        self._settings = settings

        self._scheduler = Scheduler(
            self._experiment_dao,
            self._sampling_rate_tracker,
            self._signal_objects,
            self._settings,
        )

        self._code_generator = None

    @trace("compiler.generate-code()")
    def _generate_code(self):
        code_generator = CodeGenerator(self._settings)
        self._code_generator = code_generator

        for signal_obj in self._signal_objects.values():
            code_generator.add_signal(signal_obj)

        _logger.debug("Preparing events for code generator")
        events = self._scheduler.event_timing(expand_loops=False)

        code_generator.gen_acquire_map(events)
        code_generator.gen_seq_c(
            events,
            {k: self._experiment_dao.pulse(k) for k in self._experiment_dao.pulses()},
        )
        code_generator.gen_waves()

        _logger.debug("Code generation completed")

    def run(self, near_time_parameters: Optional[ParameterStore] = None):
        self._scheduler.run(near_time_parameters)
        self._generate_code()

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
            schedule=self.prepare_schedule(),
        )

        return compiler_output

    def prepare_schedule(self):
        if not self._settings.PREPARE_PSV_DATA:
            return None

        event_list = self._scheduler.event_timing(
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=self._settings.MAX_EVENTS_TO_PUBLISH,
        )

        event_list = [
            {k: v for k, v in event.items() if v is not None} for event in event_list
        ]

        try:
            root_section = self._experiment_dao.root_rt_sections()[0]
        except IndexError:
            return {
                "event_list": [],
                "section_graph": {},
                "section_info": {},
                "subsection_map": {},
                "section_signals_with_children": {},
                "sampling_rates": [],
            }

        preorder_map = self._scheduler.preorder_map()

        section_info_out = {}

        section_signals_with_children = {}

        for section in [
            root_section,
            *self._experiment_dao.all_section_children(root_section),
        ]:
            section_info = self._experiment_dao.section_info(section)
            section_display_name = section_info.uid
            section_signals_with_children[section] = list(
                self._experiment_dao.section_signals_with_children(section)
            )
            section_info_out[section] = {
                "section_display_name": section_display_name,
                "preorder": preorder_map[section],
            }

        sampling_rate_tuples = []
        for signal_id in self._experiment_dao.signals():
            signal_info = self._experiment_dao.signal_info(signal_id)
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

        sampling_rates = [
            [list({d[0] for d in sampling_rate_tuples if d[1] == r}), r]
            for r in {t[1] for t in sampling_rate_tuples}
        ]

        _logger.debug("Pulse sheet generation completed")

        return {
            "event_list": event_list,
            "section_graph": [],  # deprecated: not needed by PSV
            "section_info": section_info_out,
            "subsection_map": {},  # deprecated: not needed by PSV
            "section_signals_with_children": section_signals_with_children,
            "sampling_rates": sampling_rates,
        }
