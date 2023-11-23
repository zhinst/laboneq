# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod

from laboneq.data.compilation_job import CompilationJob
from laboneq.data.scheduled_experiment import ScheduledExperiment


class CompilationServiceAPI(ABC):
    @abstractmethod
    def submit_compilation_job(self, job: CompilationJob):
        """
        Submit a compilation job.
        """
        raise NotImplementedError

    @abstractmethod
    def compilation_job_status(self, job_id: str):
        """
        Get the status of a compilation job.
        """
        raise NotImplementedError

    @abstractmethod
    def compilation_job_result(self, job_id: str) -> ScheduledExperiment:
        """
        Get the result of a compilation job. Blocks until the result is available.
        """
        raise NotImplementedError
