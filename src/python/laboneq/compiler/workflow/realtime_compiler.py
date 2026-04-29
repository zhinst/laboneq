# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, TypedDict

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutputContainer,
)
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.compiler_hooks import (
    get_compiler_hooks,
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
        compiler_module: compiler_rs,
        device_class: int,
        settings: CompilerSettings | None = None,
    ):
        self._experiment = experiment
        self._settings = settings if settings is not None else CompilerSettings()

        self._code_generator: ICodeGenerator = get_compiler_hooks(
            device_class
        ).code_generator()
        self._device_class = device_class
        self._compiler_module: compiler_rs = compiler_module

    def run(
        self, near_time_parameters: ParameterStore[str, float]
    ) -> RTCompilerOutputContainer:
        time_start = time.perf_counter()
        scheduler = Scheduler(compiler_module=self._compiler_module)
        schedule_result, pulse_sheet_schedule = scheduler.run(
            self._experiment, near_time_parameters
        )
        pulse_sheet_schedule = (
            _prepare_pulse_sheet_schedule(pulse_sheet_schedule)
            if pulse_sheet_schedule is not None
            else None
        )

        time_delta = time.perf_counter() - time_start
        _logger.info(f"Schedule completed. [{time_delta:.3f} s]")

        time_start = time.perf_counter()
        codegenerator = self._code_generator(schedule_result)
        codegenerator.generate_code()
        codegen_output = codegenerator.get_output()
        compiler_output = RTCompilerOutputContainer(
            device_class=self._device_class,
            codegen_output=codegen_output,
            schedule=pulse_sheet_schedule,
        )
        time_delta = time.perf_counter() - time_start
        _logger.info(f"Code generation completed for all AWGs. [{time_delta:.3f} s]")

        return compiler_output


def _prepare_pulse_sheet_schedule(schedule: compiler_rs.PulseSheetSchedule) -> Schedule:
    return Schedule(
        event_list=schedule["event_list"],
        event_list_truncated=schedule["event_list_truncated"],
        section_info=schedule["section_info"],
        section_signals_with_children=schedule["section_signals_with_children"],
        sampling_rates=schedule["sampling_rates"],
    )
