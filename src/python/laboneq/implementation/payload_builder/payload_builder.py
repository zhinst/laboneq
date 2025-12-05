# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict

from laboneq.compiler import Compiler
from laboneq.data.compilation_job import CompilationJob, ExperimentInfo
from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_description import Experiment
from laboneq.data.parameter import Parameter, SweepParameter
from laboneq.data.setup_description import Setup
from laboneq.executor.execution_from_experiment import ExecutionFactoryFromExperiment
from laboneq.implementation.payload_builder.experiment_info_builder.experiment_info_builder import (
    ExperimentInfoBuilder,
)
from laboneq.implementation.payload_builder.target_setup_generator import (
    TargetSetupGenerator,
)


def _compile(job: CompilationJob):
    compiler = Compiler(job.compiler_settings)
    compiler_output = compiler.run(job)
    compiler_output.scheduled_experiment.device_setup_fingerprint = (
        job.experiment_info.device_setup_fingerprint
    )
    return compiler_output.scheduled_experiment


class PayloadBuilder:
    def build_payload(
        self,
        device_setup: Setup,
        experiment: Experiment,
        signal_mappings: dict[str, str],
        compiler_settings: dict | None = None,
    ) -> ExecutionPayload:
        """
        Compose an experiment from a setup descriptor and an experiment descriptor.
        """
        job = self.create_compilation_job(
            device_setup, experiment, signal_mappings, compiler_settings
        )
        scheduled_experiment = _compile(job)
        target_setup = TargetSetupGenerator.from_setup(device_setup)
        run_job = ExecutionPayload(
            target_setup=target_setup,
            scheduled_experiment=scheduled_experiment,
        )
        return run_job

    def convert_to_target_setup(self, device_setup: Setup) -> TargetSetup:
        return TargetSetupGenerator.from_setup(device_setup)

    @classmethod
    def _extract_experiment_info(
        cls,
        exp: Experiment,
        setup: Setup,
        signal_mappings: dict[str, str],
    ) -> ExperimentInfo:
        builder = ExperimentInfoBuilder(exp, setup, signal_mappings)
        return builder.load_experiment()

    def create_compilation_job(
        self,
        device_setup: Setup,
        experiment: Experiment,
        signal_mappings: dict[str, str],
        compiler_settings: dict | None = None,
    ):
        experiment_info = self._extract_experiment_info(
            experiment, device_setup, signal_mappings
        )
        execution = ExecutionFactoryFromExperiment().make(
            experiment,
            driver_parameter_map=_make_driver_parameter_map(
                experiment_info.dsl_parameters
            ),
        )
        job = CompilationJob(
            experiment_info=experiment_info,
            execution=execution,
            compiler_settings=compiler_settings,
        )
        return job


def _make_driver_parameter_map(
    parameters: list[Parameter],
) -> dict[str, list[Parameter]]:
    """Make a mapping from driver UIDs to the parameters they drive."""
    driver_map: dict[str, set[str]] = defaultdict(set)

    def collect_driving_parameters(parameter: SweepParameter) -> set[str]:
        drivers = set()
        if not isinstance(parameter, SweepParameter):
            return drivers
        for driver in parameter.driven_by:
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
