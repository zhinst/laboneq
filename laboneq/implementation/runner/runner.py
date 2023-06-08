# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from random import random
from threading import Timer

from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.interfaces.runner.runner_api import RunnerAPI

_logger = logging.getLogger(__name__)


class Runner(RunnerAPI):
    """
    This the core implementation of the experiment runner.
    """

    def __init__(self):
        self._job_queue = []
        self._job_results = {}

    def submit_execution_payload(self, job: ExecutionPayload):
        """
        Submit an experiment run job.
        """
        job_id = len(self._job_queue)
        queue_entry = {"job_id": job_id, "job": job}

        def complete_job():
            acquired_results = {k: random() for k in job.recipe.measurement_map.keys()}
            results = ExperimentResults(acquired_results=acquired_results)
            self._job_results[job_id] = results

        Timer(1, complete_job).start()
        self._job_queue.append(queue_entry)
        return job_id

    def run_job_status(self, job_id: str):
        """
        Get the status of an  experiment run job.
        """
        return next(j for j in self._job_queue if j["job_id"] == job_id)

    def run_job_result(self, job_id: str) -> ExperimentResults:
        """
        Get the result of an experiment run job. Blocks until the result is available.
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

    def connect(self, setup: TargetSetup, do_emulation: bool = True):
        """
        Connect to the setup
        """
        pass

    def start(self):
        """
        Start the experiment runner. It will start processing jobs from the job queue.
        """
        pass

    def stop(self):
        """
        Stop the experiment runner. It will stop processing jobs from the job queue.
        """
        pass

    def disconnect(self):
        """
        Disconnect from the setup.
        """
        pass
