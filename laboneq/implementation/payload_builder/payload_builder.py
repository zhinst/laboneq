# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
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
from laboneq.interfaces.compilation_service.compilation_service_api import (
    CompilationServiceAPI,
)
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI


class PayloadBuilder(PayloadBuilderAPI):
    def __init__(self, compilation_service: CompilationServiceAPI = None):
        self._compilation_service: CompilationServiceAPI = compilation_service

    def build_payload(
        self,
        device_setup: Setup,
        experiment: Experiment,
        signal_mappings: dict[str, str],
    ) -> ExecutionPayload:
        """
        Compose an experiment from a setup descriptor and an experiment descriptor.
        """

        experiment = copy.deepcopy(experiment)
        if experiment.signals is None:
            experiment.signals = []

        experiment_info = self._extract_experiment_info(
            experiment, device_setup, signal_mappings
        )
        execution = ExecutionFactoryFromNewExperiment().make(experiment)
        job = CompilationJob(experiment_info=experiment_info, execution=execution)
        job_id = self._compilation_service.submit_compilation_job(job)

        scheduled_experiment = self._compilation_service.compilation_job_result(job_id)

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
