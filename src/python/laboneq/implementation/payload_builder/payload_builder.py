# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

from laboneq.compiler import Compiler
from laboneq.data.compilation_job import CompilationJob
from laboneq.data.parameter import SweepParameter as DataSweepParameter
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.data.setup_description import Setup
from laboneq.dsl.experiment import Experiment
from laboneq.dsl.parameter import Parameter, SweepParameter
from laboneq.executor.execution_from_experiment import ExecutionFactoryFromExperiment
from laboneq.implementation.payload_builder.experiment_info_builder.experiment_info_builder import (
    ExperimentInfoBuilder,
)


def compile_experiment(
    device_setup: Setup,
    experiment: Experiment,
    signal_mappings: dict[str, str],
    compiler_settings: dict | None = None,
) -> ScheduledExperiment:
    experiment_info = ExperimentInfoBuilder(
        experiment, device_setup, signal_mappings
    ).load_experiment()
    execution = ExecutionFactoryFromExperiment().make(
        experiment,
        driver_parameter_map=_make_driver_parameter_map(experiment_info.dsl_parameters),
    )
    compiler = Compiler(compiler_settings)
    return compiler.run(
        CompilationJob(experiment_info=experiment_info, execution=execution)
    )


def _make_driver_parameter_map(
    parameters: list[Parameter],
) -> dict[str, list[Parameter]]:
    """Make a mapping from driver UIDs to the parameters they drive."""
    driver_map: dict[str, set[str]] = defaultdict(set)

    def collect_driving_parameters(parameter: Parameter) -> set[str]:
        drivers = set()
        # TODO(DSL cutover): Remove DataSweepParameter once setup calibration uses DSL types.
        if not isinstance(parameter, (SweepParameter, DataSweepParameter)):
            return drivers
        for driver in parameter.driven_by or []:
            driver_map[driver.uid].add(parameter.uid)
            drivers.add(driver.uid)
            drivers.update(collect_driving_parameters(driver))
        return drivers

    for param in parameters:
        drivers = collect_driving_parameters(param)
        for driver_uid in drivers:
            driver_map[driver_uid].add(param.uid)
    param_map = {p.uid: p for p in parameters}
    return {k: [param_map[uid] for uid in v] for k, v in driver_map.items()}
