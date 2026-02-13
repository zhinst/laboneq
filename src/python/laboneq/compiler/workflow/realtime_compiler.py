# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import logging
import time
from itertools import groupby
from typing import TypedDict

from laboneq._rust import compiler as compiler_rs
from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.iface_code_generator import ICodeGenerator
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutput,
    RTCompilerOutputContainer,
)
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.scheduler.parameter_store import ParameterStore
from laboneq.compiler.scheduler.sampling_rate_tracker import SamplingRateTracker
from laboneq.compiler.scheduler.scheduler import Scheduler
from laboneq.compiler.workflow.compiler_hooks import (
    all_compiler_hooks,
    get_compiler_hooks,
)
from laboneq.data.compilation_job import SignalInfo

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
        settings: CompilerSettings | None = None,
    ):
        self._signal_infos = signal_infos
        self._experiment = experiment
        self._sampling_rate_tracker = sampling_rate_tracker
        self._signal_objects = signal_objects
        self._settings = settings if settings is not None else CompilerSettings()

        self._code_generators: dict[int, ICodeGenerator] = {}
        self._rust_experiment_ir: compiler_rs.ExperimentIr | None = None

    def _lower_ir_to_code(self, ir_rust: compiler_rs.ExperimentIr):
        awgs = [signal_obj.awg for signal_obj in self._signal_objects.values()]
        device_classes = {awg.device_class for awg in awgs}
        known_device_classes = set(h.device_class() for h in all_compiler_hooks())
        unknown_devices = [
            awg for awg in awgs if awg.device_class not in known_device_classes
        ]

        if len(unknown_devices) != 0:
            raise Exception("Invalid device class encountered")

        for device_class in device_classes:
            signals = [
                s
                for s in self._signal_objects.values()
                if s.awg.device_class == device_class
            ]
            self._code_generators[device_class] = get_compiler_hooks(
                device_class
            ).code_generator()(
                ir_rust,
                settings=self._settings,
                signals=signals,
            )
            self._code_generators[device_class].generate_code()

        _logger.debug("Code generation completed")

    def run(
        self, near_time_parameters: ParameterStore[str, float]
    ) -> RTCompilerOutputContainer:
        time_start = time.perf_counter()
        scheduler = Scheduler()
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

    def _lower_ir_to_pulse_sheet(self):
        rust_schedule = compiler_rs.generate_schedule(
            self._rust_experiment_ir,
            expand_loops=self._settings.EXPAND_LOOPS_FOR_SCHEDULE,
            max_events=int(self._settings.MAX_EVENTS_TO_PUBLISH),
        )

        # Extract sampling rates directly from self._signal_infos
        sampling_rate_tuples = []
        for signal_info in self._signal_infos:
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
        # Also sort the device lists for consistent output.
        sampling_rates = [
            (sorted({tpl[0] for tpl in grouped_tuples}), sampling_rate)
            for sampling_rate, grouped_tuples in groupby(
                sampling_rate_tuples, lambda tpl: tpl[1]
            )
        ]

        _logger.debug("Pulse sheet generation completed")

        return Schedule(
            event_list=rust_schedule["event_list"],
            section_info=rust_schedule["section_info"],
            section_signals_with_children=rust_schedule[
                "section_signals_with_children"
            ],
            sampling_rates=sampling_rates,
        )

    def prepare_schedule(self) -> Schedule:
        return self._lower_ir_to_pulse_sheet()
