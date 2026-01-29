# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import copy
import logging
import time
from itertools import groupby
from typing import TYPE_CHECKING, TypedDict

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutput,
    RTCompilerOutputContainer,
)
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.event_list.event_list_generator import generate_event_list_from_ir
from laboneq.compiler.event_list.preorder_map import preorder_map
from laboneq.compiler.feedback_router.feedback_router import FeedbackRegisterLayout
from laboneq.compiler.ir.ir import IRTree
from laboneq.compiler.ir.section_ir import SectionIR
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.compiler_hooks import (
    all_compiler_hooks,
    get_compiler_hooks,
)
from laboneq.data.compilation_job import SignalInfo

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs

_logger = logging.getLogger(__name__)


class Schedule(TypedDict):
    event_list: list[dict]
    section_info: dict[str, dict]
    section_signals_with_children: dict[str, list[str]]
    sampling_rates: list[tuple[list[str], float]]


class RealtimeCompiler:
    def __init__(
        self,
        experiment: compiler_rs.ExperimentInfo,
        signal_infos: list[SignalInfo],
        sampling_rate_tracker: SamplingRateTracker,
        signal_objects: dict[str, SignalObj],
        feedback_register_layout: FeedbackRegisterLayout,
        settings: CompilerSettings | None = None,
    ):
        self._signal_infos = signal_infos
        self._experiment = experiment
        self._feedback_register_layout = feedback_register_layout
        self._sampling_rate_tracker = sampling_rate_tracker
        self._signal_objects = signal_objects
        self._settings = settings if settings is not None else CompilerSettings()

        self._code_generators: dict[int, ICodeGenerator] = {}

    def _lower_ir_to_code(self, ir: IRTree):
        awgs = [signal_obj.awg for signal_obj in self._signal_objects.values()]
        device_classes = {awg.device_class for awg in awgs}
        known_device_classes = set(h.device_class() for h in all_compiler_hooks())
        unknown_devices = [
            awg for awg in awgs if awg.device_class not in known_device_classes
        ]

        if len(unknown_devices) != 0:
            raise Exception("Invalid device class encountered")

        # The backends mutate the IR as they lower it. If we use more than 1, we
        # must provide each with a pristine copy of the original IR.
        maybe_copy = copy.deepcopy if len(device_classes) > 1 else lambda x: x

        for device_class in device_classes:
            signals = [
                s
                for s in self._signal_objects.values()
                if s.awg.device_class == device_class
            ]
            self._code_generators[device_class] = get_compiler_hooks(
                device_class
            ).code_generator()(
                maybe_copy(ir),
                settings=self._settings,
                signals=signals,
                feedback_register_layout=self._feedback_register_layout,
            )
            self._code_generators[device_class].generate_code()

        _logger.debug("Code generation completed")

    def run(
        self, near_time_parameters: ParameterStore[str, float]
    ) -> RTCompilerOutputContainer:
        time_start = time.perf_counter()
        scheduler = Scheduler()
        scheduler.run(self._experiment, near_time_parameters)
        ir = scheduler.generate_ir(
            pulse_defs=self._experiment.pulse_defs, signals=self._signal_infos
        )
        schedule = self.prepare_schedule(ir) if self._settings.OUTPUT_EXTRAS else None
        time_delta = time.perf_counter() - time_start
        _logger.info(f"Schedule completed. [{time_delta:.3f} s]")

        time_start = time.perf_counter()
        self._lower_ir_to_code(ir)
        outputs: dict[int, RTCompilerOutput] = {
            device_class: code_generator.get_output()
            for device_class, code_generator in self._code_generators.items()
        }
        compiler_output = RTCompilerOutputContainer(
            codegen_output=outputs, schedule=schedule
        )
        time_delta = time.perf_counter() - time_start
        _logger.info(f"Code generation completed for all AWGs. [{time_delta:.3f} s]")

        return compiler_output

    def _lower_ir_to_pulse_sheet(self, ir: IRTree):
        event_list = generate_event_list_from_ir(
            ir=ir,
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=self._settings.MAX_EVENTS_TO_PUBLISH,
        )
        event_list = [
            {k: v for k, v in event.items() if v is not None} for event in event_list
        ]

        def find_root_section(ir_node):
            if isinstance(ir_node, SectionIR):
                return ir_node
            for child in ir_node.children:
                if (r := find_root_section(child)) is not None:
                    return r
            return None

        root_section: SectionIR | None = find_root_section(ir.root)
        assert root_section is not None
        preorder_map_ = preorder_map(root_section)

        section_info_out = {}
        section_signals_with_children = {}

        def all_sections(section_ir):
            assert isinstance(section_ir, SectionIR)
            yield section_ir
            for child in section_ir.children:
                if isinstance(child, SectionIR):
                    yield from (all_sections(child))

        for section_ir in all_sections(root_section):
            try:  # pulse may be missing in case of experiments using match case
                section_id = section_ir.section
                section_display_name = section_id
                section_signals_with_children[section_id] = list(section_ir.signals)
                section_info_out[section_id] = {
                    "section_display_name": section_display_name,
                    "preorder": preorder_map_[section_id],
                }
            except KeyError:  # noqa: PERF203
                continue

        sampling_rate_tuples = []
        for signal_info in ir.signals:
            assert signal_info.device is not None
            device_id = signal_info.device.uid
            assert signal_info.device.device_type is not None
            device_type = signal_info.device.device_type.value
            sampling_rate_tuples.append(
                (
                    device_type,
                    int(
                        self._sampling_rate_tracker.sampling_rate_for_device(
                            device_id, signal_info.type
                        )
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

    def prepare_schedule(self, ir: IRTree) -> Schedule:
        return self._lower_ir_to_pulse_sheet(ir)
