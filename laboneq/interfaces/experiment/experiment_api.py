# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict

from laboneq.data.execution_payload import ExecutionPayload
from laboneq.data.experiment_description import Experiment
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.setup_description import Setup


class ExperimentAPI(ABC):
    @abstractmethod
    def load_setup(
        self, setup_descriptor, server_host=None, server_port=None, setup_name=None
    ):
        """
        Load a setup from a descriptor.
        """
        raise NotImplementedError

    @abstractmethod
    def current_setup(self):
        """
        Get the current setup.
        """
        raise NotImplementedError

    @abstractmethod
    def new_experiment(self) -> Experiment:
        """
        Create a new experiment
        """
        raise NotImplementedError

    @abstractmethod
    def current_experiment(self) -> Experiment:
        """
        Get the current experiment
        """
        raise NotImplementedError

    @abstractmethod
    def run_current_experiment(
        self, setup: Setup, signal_mappings: Dict[str, str]
    ) -> ExperimentResults:
        """
        Run the current experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def run_payload(self, execution_payload: ExecutionPayload):
        """
        Run an experiment job.
        """
        raise NotImplementedError

    @abstractmethod
    def set_current_experiment(self, experiment: Experiment):
        """
        Set the current experiment.
        """
        raise NotImplementedError

    @abstractmethod
    def set_current_setup(self, setup: Setup):
        """
        Set the current setup.
        """
        raise NotImplementedError

    @abstractmethod
    def build_payload_for_current_experiment(
        self, compiler_settings: dict | None = None
    ) -> ExecutionPayload:
        """
        Compose the current experiment with a setup.
        """

        raise NotImplementedError

    @abstractmethod
    def map_signals(self, signal_mappings: Dict[str, str]):
        """
        Map experiment signals to logical signals.
        """
        raise NotImplementedError
