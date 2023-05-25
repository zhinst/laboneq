# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.interfaces.runner.runner_api import RunnerAPI
from laboneq.interfaces.runner.runner_control_api import RunnerControlAPI

_logger = logging.getLogger(__name__)


class Runner(RunnerAPI, RunnerControlAPI):
    """
    This the core implementation of the experiment runner.
    """

    # Currently, just a dummy implementation

    def __init__(self):
        pass

    def submit_execution_payload(self, job: ExecutionPayload):
        """
        Submit an experiment run job.
        """

        return None

    def run_job_status(self, job_id: str):
        """
        Get the status of an  experiment run job.
        """
        return None

    def run_job_result(self, job_id: str) -> ExperimentResults:
        """
        Get the result of an experiment run job. Blocks until the result is available.
        """
        return None

    def connect(self, setup: TargetSetup, do_emulation: bool = True):
        """
        Connect to the setup
        """
        return None

    def start(self):
        """
        Start the experiment runner. It will start processing jobs from the job queue.
        """
        return None

    def stop(self):
        """
        Stop the experiment runner. It will stop processing jobs from the job queue.
        """
        return None

    def disconnect(self):
        """
        Disconnect from the setup.
        """
        return None
