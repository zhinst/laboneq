# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import logging
from typing import Dict

from laboneq.data.execution_payload import ExecutionPayload
from laboneq.data.experiment_description import Experiment
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.setup_description import Setup
from laboneq.implementation.experiment_workflow.device_setup_generator import (
    DeviceSetupGenerator,
)
from laboneq.interfaces.application_management.laboneq_settings import LabOneQSettings
from laboneq.interfaces.experiment.experiment_api import ExperimentAPI
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI
from laboneq.interfaces.runner.runner_api import RunnerAPI
from laboneq.interfaces.runner.runner_control_api import RunnerControlAPI

_logger = logging.getLogger(__name__)


class ExperimentWorkflow(ExperimentAPI):
    def __init__(
        self,
        runner: RunnerAPI,
        payload_builder: PayloadBuilderAPI,
        runner_control: RunnerControlAPI,
        settings: LabOneQSettings,
    ):
        self._current_setup = None
        self._experiment_counter = 0
        self._runner = runner
        self._payload_builder = payload_builder
        self._runner_control = runner_control
        self._settings = settings
        self._current_experiment = None
        self._signal_mappings = None

        if id(self._runner_control) != id(self._runner):
            raise ValueError("RunnerControl and Runner must be the same object")

    def load_setup(
        self, setup_descriptor, server_host=None, server_port=None, setup_name=None
    ):
        """
        Load a setup from a descriptor.
        """
        self._current_setup = self.device_setup_from_descriptor(
            setup_descriptor, server_host, server_port, setup_name
        )

    def current_setup(self):
        """
        Get the current setup.
        """
        return copy.deepcopy(self._current_setup)

    def new_experiment(self) -> Experiment:
        """
        Create a new experiment
        """
        self._current_experiment = Experiment(uid="exp" + str(self._experiment_counter))
        self._experiment_counter += 1
        return copy.deepcopy(self._current_experiment)

    def current_experiment(self):
        """
        Get the current experiment
        """
        return copy.deepcopy(self._current_experiment)

    def run_current_experiment(
        self, setup: Setup, signal_mappings: Dict[str, str]
    ) -> ExperimentResults:
        """
        Run the current experiment.
        """

        self._signal_mappings = signal_mappings
        self.set_current_setup(setup)
        execution_payload: ExecutionPayload = (
            self.build_payload_for_current_experiment()
        )

        job_id = self._runner.submit_execution_payload(execution_payload)
        return self._runner.run_job_result(job_id)

    def run_payload(self, execution_payload: ExecutionPayload):
        """
        Run an experiment job.
        """
        job_id = self._runner.submit_execution_payload(execution_payload)
        return self._runner.run_job_result(job_id)

    def build_payload_for_current_experiment(
        self, compiler_settings: dict | None = None
    ) -> ExecutionPayload:
        """
        Compose the current experiment with a setup.
        """

        if self._signal_mappings is None:
            raise ValueError(
                "Signal mappings must be set before building payload for experiment"
            )
        execution_payload = self._payload_builder.build_payload(
            self._current_setup,
            self._current_experiment,
            self._signal_mappings,
            compiler_settings,
        )

        return copy.deepcopy(execution_payload)

    def set_current_experiment(self, experiment: Experiment):
        """
        Set the current experiment.
        """
        self._current_experiment = copy.deepcopy(experiment)

    def device_setup_from_descriptor(
        self,
        yaml_text: str,
        server_host: str | None = None,
        server_port: str | int | None = None,
        setup_name: str | None = None,
    ) -> Setup:
        """
        Create a device setup from a descriptor.
        """
        return DeviceSetupGenerator.from_descriptor(
            yaml_text, server_host, server_port, setup_name
        )

    def set_current_setup(self, setup: Setup):
        """
        Set the current setup.
        """
        self._current_setup = copy.deepcopy(setup)
        if self._settings.runner_is_local:
            _logger.debug(f"Experiment runner is local, connecting to {setup.uid}")
            target_setup = self._payload_builder.convert_to_target_setup(setup)
            self._runner_control.connect(target_setup)
            # in local mode, we start the experiment runner immediately
            self._runner_control.start()
        # TODO: in remote mode, we need to find the correct queue from the setup
        # so that we can then submit jobs to that queue.

    def map_signals(self, signal_mappings: Dict[str, str]):
        """
        Map experiment signals to logical signals.
        """
        self._signal_mappings = {}
        _logger.debug(
            f"Mapping signals, experiment signals: {self._current_experiment.signals}"
        )
        experiment_signals_by_uid = {
            es.uid: es for es in self._current_experiment.signals
        }
        for k, v in signal_mappings.items():
            experiment_signal = experiment_signals_by_uid[k]
            *_, grp, ls_name = v.split("/")

            if grp in self._current_setup.logical_signal_groups:
                if (
                    ls_name
                    in self._current_setup.logical_signal_groups[grp].logical_signals
                ):
                    self._signal_mappings[experiment_signal.uid] = v
