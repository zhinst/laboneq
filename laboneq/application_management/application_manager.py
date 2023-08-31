# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass

from lagom import Container, Singleton

from laboneq.implementation.compilation_service.compilation_service_legacy import (
    CompilationServiceLegacy,
)
from laboneq.implementation.experiment_workflow import ExperimentWorkflow
from laboneq.implementation.payload_builder.payload_builder import PayloadBuilder
from laboneq.implementation.runner.runner_legacy import RunnerLegacy
from laboneq.interfaces.application_management.laboneq_settings import LabOneQSettings
from laboneq.interfaces.compilation_service.compilation_service_api import (
    CompilationServiceAPI,
)
from laboneq.interfaces.experiment.experiment_api import ExperimentAPI
from laboneq.interfaces.payload_builder.payload_builder_api import PayloadBuilderAPI
from laboneq.interfaces.runner.runner_api import RunnerAPI
from laboneq.interfaces.runner.runner_control_api import RunnerControlAPI

_logger = logging.getLogger(__name__)


@dataclass(init=False)
class LaboneQDefaultSettings(LabOneQSettings):
    def __init__(self):
        self.runner_is_local = True
        self.compilation_service_is_local = True

    # TODO: use  a configration file or environment variables to set these values
    # Maybe use dotenv (see https://pypi.org/project/python-dotenv/)
    # and/or dynaconf (see https://www.dynaconf.com/)
    runner_is_local: bool
    compilation_service_is_local: bool


class ApplicationManager:
    _instance = None

    def __init__(self):
        self._experimenter_api = None
        self._payload_builder = None

    def start(self):
        if self._experimenter_api is not None:
            _logger.warning("ApplicationManager already started.")
            return
        container = Container(log_undefined_deps=True)
        container[LabOneQSettings] = LaboneQDefaultSettings
        container[RunnerControlAPI] = Singleton(RunnerLegacy)
        # RunnerControlAPI and the RunnerAPI are currently implemented by the same object:
        container[RunnerAPI] = container[RunnerControlAPI]
        container[CompilationServiceAPI] = CompilationServiceLegacy
        container[PayloadBuilderAPI] = PayloadBuilder
        container[ExperimentAPI] = ExperimentWorkflow
        self._experimenter_api = container[ExperimentAPI]
        self._payload_builder = container[PayloadBuilderAPI]

    def laboneq(self) -> ExperimentAPI:
        return self._experimenter_api

    @staticmethod
    def instance() -> "ApplicationManager":
        if ApplicationManager._instance is None:
            ApplicationManager._instance = ApplicationManager()
            ApplicationManager._instance.start()
        return ApplicationManager._instance
