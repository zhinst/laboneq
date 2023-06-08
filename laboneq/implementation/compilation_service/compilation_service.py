# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.data.compilation_job import CompilationJob
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.interfaces.compilation_service.compilation_service_api import (
    CompilationServiceAPI,
)

_logger = logging.getLogger(__name__)


class CompilationService(CompilationServiceAPI):
    """
    This the core implementation of the compilation service.
    """

    def __init__(self):
        pass

    def submit_compilation_job(self, job: CompilationJob):
        """
        Submit a compilation job.
        """

        return None

    def compilation_job_status(self, job_id: str):
        """
        Get the status of a compilation job.
        """
        return None

    def compilation_job_result(self, job_id: str) -> ScheduledExperiment:
        """
        Get the result of a compilation job. Blocks until the result is available.
        """

        return ScheduledExperiment(
            recipe={
                "experiment": {"initializations": [], "realtime_execution_init": []}
            }
        )
