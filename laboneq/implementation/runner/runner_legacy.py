# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
import time

from laboneq import controller as ctrl
from laboneq.data.execution_payload import ExecutionPayload, TargetSetup
from laboneq.data.experiment_results import ExperimentResults
from laboneq.interfaces.runner.runner_api import RunnerAPI
from laboneq.interfaces.runner.runner_control_api import RunnerControlAPI

_logger = logging.getLogger(__name__)


class RunnerLegacy(RunnerAPI, RunnerControlAPI):
    """
    This the core implementation of the experiment runner.
    """

    def __init__(self):
        self._job_queue = []
        self._job_results = {}
        self._connected = False
        self._controller = None

    def connect(self, setup: TargetSetup, do_emulation: bool = True):
        _logger.debug(f"Connecting to TargetSetup {setup.uid}")
        run_parameters = ctrl.ControllerRunParameters()
        run_parameters.dry_run = do_emulation
        run_parameters.ignore_version_mismatch = do_emulation

        controller = ctrl.Controller(
            run_parameters=run_parameters,
            target_setup=setup,
            neartime_callbacks={},
        )
        controller.connect()
        self._controller = controller
        self._connected = True

    def submit_execution_payload(self, job: ExecutionPayload):
        """
        Submit an experiment run job.
        """
        job_id = len(self._job_queue)
        queue_entry = {"job_id": job_id, "job": job}

        self._job_queue.append(queue_entry)
        if not self._connected:
            self.connect(job.target_setup)

        self._controller.execute_compiled(job)
        controller_results = self._controller._results
        self._job_results[job_id] = ExperimentResults(
            acquired_results=controller_results.acquired_results,
            neartime_callback_results=controller_results.neartime_callback_results,
            execution_errors=controller_results.execution_errors,
        )

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
