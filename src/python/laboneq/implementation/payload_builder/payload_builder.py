# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import uuid

from laboneq.data.compilation_job import CompilationJob, ExperimentInfo
from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_description import Experiment
from laboneq.data.setup_description import Setup
from laboneq.executor.execution_from_new_experiment import (
    ExecutionFactoryFromNewExperiment,
)
from laboneq.implementation.payload_builder.experiment_info_builder.experiment_info_builder import (
    ExperimentInfoBuilder,
)
from laboneq.implementation.payload_builder.target_setup_generator import (
    TargetSetupGenerator,
)
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI
from laboneq.compiler import Compiler


def _compile(job: CompilationJob):
    compiler = Compiler(job.compiler_settings)
    compiler_output = compiler.run(job)
    compiler_output.scheduled_experiment.uid = uuid.uuid4().hex
    return compiler_output.scheduled_experiment


class PayloadBuilder(PayloadBuilderAPI):
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
            uid=uuid.uuid4().hex,
            target_setup=target_setup,
            compiled_experiment_hash=scheduled_experiment.uid,
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
        execution = ExecutionFactoryFromNewExperiment().make(experiment)
        job = CompilationJob(
            experiment_info=experiment_info,
            execution=execution,
            compiler_settings=compiler_settings,
        )
        return job
