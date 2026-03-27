# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, TypedDict

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutput,
    RTCompilerOutputContainer,
)
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.compiler_hooks import (
    get_compiler_hooks,
    resolve_compiler_module,
)

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs

_logger = logging.getLogger(__name__)


class Schedule(TypedDict):
    event_list: list[dict]
    event_list_truncated: bool
    section_info: dict[str, dict]
    section_signals_with_children: dict[str, list[str]]
    sampling_rates: list[tuple[list[str], float]]


class RealtimeCompiler:
    def __init__(
        self,
        experiment,
        signal_objects: dict[str, SignalObj],
        settings: CompilerSettings | None = None,
    ):
        self._experiment = experiment
        self._signal_objects = signal_objects
        self._settings = settings if settings is not None else CompilerSettings()

        self._code_generators: dict[int, ICodeGenerator] = {}
        self._rust_experiment_ir = None
        self._compiler_module: compiler_rs = resolve_compiler_module(
            {s.awg.device_class for s in signal_objects.values()}
        )

    def _lower_ir_to_code(self, ir_rust):
        awgs = [signal_obj.awg for signal_obj in self._signal_objects.values()]
        device_classes = {awg.device_class for awg in awgs}
        for device_class in device_classes:
            self._code_generators[device_class] = get_compiler_hooks(
                device_class
            ).code_generator()(
                ir_rust,
                settings=self._settings,
            )
            self._code_generators[device_class].generate_code()

        _logger.debug("Code generation completed")

    def run(
        self, near_time_parameters: ParameterStore[str, float]
    ) -> RTCompilerOutputContainer:
        time_start = time.perf_counter()
        scheduler = Scheduler(compiler_module=self._compiler_module)
        schedule_result = scheduler.run(self._experiment, near_time_parameters)
        # Store the Rust ExperimentIr for schedule generation
        self._rust_experiment_ir = schedule_result
        schedule = self.prepare_schedule() if self._settings.OUTPUT_EXTRAS else None
        time_delta = time.perf_counter() - time_start
        _logger.info(f"Schedule completed. [{time_delta:.3f} s]")

        time_start = time.perf_counter()
        self._lower_ir_to_code(schedule_result)
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

    def prepare_schedule(self) -> Schedule:
        rust_schedule = self._compiler_module.generate_pulse_sheet_schedule(
            self._rust_experiment_ir,
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=int(self._settings.MAX_EVENTS_TO_PUBLISH),
        )
        return Schedule(
            event_list=rust_schedule["event_list"],
            event_list_truncated=rust_schedule["event_list_truncated"],
            section_info=rust_schedule["section_info"],
            section_signals_with_children=rust_schedule[
                "section_signals_with_children"
            ],
            sampling_rates=rust_schedule["sampling_rates"],
        )
