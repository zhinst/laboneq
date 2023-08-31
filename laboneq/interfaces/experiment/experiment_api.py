# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Dict

from laboneq.data.execution_payload import ExecutionPayload
from laboneq.data.experiment_description import Experiment
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.setup_description import Setup


class ExperimentAPI(ABC):
    def load_setup(
        self, setup_descriptor, server_host=None, server_port=None, setup_name=None
    ):
        """
        Load a setup from a descriptor.
        """
        raise NotImplementedError

    def current_setup(self):
        """
        Get the current setup.
        """
        raise NotImplementedError

    def new_experiment(self) -> Experiment:
        """
        Create a new experiment
        """
        raise NotImplementedError

    def current_experiment(self) -> Experiment:
        """
        Get the current experiment
        """
        raise NotImplementedError

    def run_current_experiment(
        self, setup: Setup, signal_mappings: Dict[str, str]
    ) -> ExperimentResults:
        """
        Run the current experiment.
        """
        raise NotImplementedError

    def run_payload(self, execution_payload: ExecutionPayload):
        """
        Run an experiment job.
        """
        raise NotImplementedError

    def set_current_experiment(self, experiment: Experiment):
        """
        Set the current experiment.
        """
        raise NotImplementedError

    def set_current_setup(self, setup: Setup):
        """
        Set the current setup.
        """
        raise NotImplementedError

    def build_payload_for_current_experiment(
        self, compiler_settings: dict = None
    ) -> ExecutionPayload:
        """
        Compose the current experiment with a setup.
        """

        raise NotImplementedError

    def map_signals(self, signal_mappings: Dict[str, str]):
        """
        Map experiment signals to logical signals.
        """
        raise NotImplementedError
