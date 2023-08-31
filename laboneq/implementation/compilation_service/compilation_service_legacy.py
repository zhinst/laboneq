# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import time
import uuid

from laboneq.compiler import Compiler
from laboneq.core.types.compiled_experiment import (
    CompiledExperiment as CompiledExperimentDSL,
)
from laboneq.data.compilation_job import CompilationJob
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.interfaces.compilation_service import CompilationServiceAPI


class CompilationServiceLegacy(CompilationServiceAPI):
    def __init__(self):
        self._job_queue = []
        self._job_results = {}

    def submit_compilation_job(self, job: CompilationJob):
        """
        Submit a compilation job.
        """
        job_id = len(self._job_queue)
        queue_entry = {"job_id": job_id, "job": job}
        compiler = Compiler(job.compiler_settings)
        compiler_output = compiler.run(job)

        self._job_results[job_id] = convert_compiler_output_to_scheduled_experiment(
            compiler_output
        )

        self._job_queue.append(queue_entry)
        return job_id

    def compilation_job_status(self, job_id: str):
        """
        Get the status of a compilation job.
        """
        return next(j for j in self._job_queue if j["job_id"] == job_id)

    def compilation_job_result(self, job_id: str) -> ScheduledExperiment:
        """
        Get the result of a compilation job. Blocks until the result is available.
        """
        num_tries = 10
        while True:
            result = self._job_results.get(job_id)
            if result:
                return result
            if num_tries == 0:
                break
            num_tries -= 1
            time.sleep(100e-3)


def convert_compiler_output_to_scheduled_experiment(
    compiler_output: CompiledExperimentDSL,
) -> ScheduledExperiment:
    scheduled_experiment = copy.deepcopy(compiler_output.scheduled_experiment)
    scheduled_experiment.uid = uuid.uuid4().hex

    return scheduled_experiment
