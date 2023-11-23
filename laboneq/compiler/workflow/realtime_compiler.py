# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
from itertools import groupby
from typing import Dict, Optional, TypedDict

from laboneq._observability.tracing import trace
from laboneq.compiler import CodeGenerator, CompilerSettings
from laboneq.compiler.code_generator.ir_to_event_list import generate_event_list_from_ir
from laboneq.compiler.code_generator.code_generator_pretty_printer import PrettyPrinter
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.experiment_access import ExperimentDAO
from laboneq.compiler.ir.ir import IR
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.compiler_output import RealtimeCompilerOutput

_logger = logging.getLogger(__name__)

_registered_codegens = {
    0x0: CodeGenerator,
    0x1: PrettyPrinter,
}


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


class RealtimeCompiler:
    def __init__(
        self,
        experiment_dao: ExperimentDAO,
        scheduler: Scheduler,
        sampling_rate_tracker: SamplingRateTracker,
        signal_objects: Dict[str, SignalObj],
        settings: CompilerSettings | None = None,
    ):
        self._experiment_dao = experiment_dao
        self._ir = None
        self._scheduler = scheduler
        self._sampling_rate_tracker = sampling_rate_tracker
        self._signal_objects = signal_objects
        self._settings = settings

        self._code_generators = {}

    def _lower_to_ir(self):
        return self._scheduler.generate_ir()

    def _lower_ir_to_code(self, ir: IR):
        if len(self._signal_objects) == 0:
            self._code_generators[0] = CodeGenerator(settings=self._settings, ir=ir)
            self._code_generators[0].generate_code(self._signal_objects)
            return

        awgs = [signal_obj.awg for signal_obj in self._signal_objects.values()]
        device_classes = {awg.device_class for awg in awgs}
        unknown_devices = [
            awg for awg in awgs if awg.device_class not in _registered_codegens
        ]

        if len(unknown_devices) != 0:
            raise Exception("Invalid device class encountered")

        for device_class in device_classes:
            self._code_generators[device_class] = _registered_codegens[device_class](
                ir, settings=self._settings
            )
            self._code_generators[device_class].generate_code(
                [
                    s
                    for s in self._signal_objects.values()
                    if s.awg.device_class == device_class
                ]
            )

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

        outputs = {
            device_class: code_generator.fill_output()
            for device_class, code_generator in self._code_generators.items()
        }

        compiler_output = RealtimeCompilerOutput(
            codegen_output=outputs,
            schedule=schedule,
        )

        return compiler_output

    def _lower_ir_to_schedule(self, ir: IR):
        event_list = generate_event_list_from_ir(
            ir=ir,
            settings=self._settings,
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=self._settings.MAX_EVENTS_TO_PUBLISH,
        )

        event_list = [
            {k: v for k, v in event.items() if v is not None} for event in event_list
        ]

        try:
            root_section = self._experiment_dao.root_rt_sections()[0]
        except IndexError:
            return Schedule.empty()

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

        # Group devices by sampling rate and create a backward compatible list of those.
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
