# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.compiler import Compiler
from laboneq.data.compilation_job import CompilationJob
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.data.setup_description import Setup
from laboneq.dsl.experiment import Experiment
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

    dsl_parameters_map = {param.uid: param for param in experiment_info.dsl_parameters}
    driver_parameter_map = {
        driver: [dsl_parameters_map[uid] for uid in driven_uids]
        for driver, driven_uids in experiment_info.driving_parameters.items()
    }

    execution = ExecutionFactoryFromExperiment().make(
        experiment,
        driver_parameter_map=driver_parameter_map,
    )
    compiler = Compiler(compiler_settings)
    return compiler.run(
        CompilationJob(experiment_info=experiment_info, execution=execution)
    )
