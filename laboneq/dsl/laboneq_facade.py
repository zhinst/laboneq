# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING

from numpy import typing as npt

from laboneq import controller as ctrl
from laboneq.application_management.application_manager import ApplicationManager
from laboneq.core.types import CompiledExperiment
from laboneq.implementation.legacy_adapters.converters_experiment_description import (
    convert_Experiment,
    convert_signal_map,
)
from laboneq.implementation.legacy_adapters.converters_target_setup import (
    convert_dsl_to_target_setup,
)
from laboneq.implementation.legacy_adapters.device_setup_converter import (
    convert_device_setup_to_setup,
)

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.session import Session


class LabOneQFacade:
    @staticmethod
    def connect(session: Session):
        run_parameters = ctrl.ControllerRunParameters()
        run_parameters.dry_run = session._connection_state.emulated
        run_parameters.ignore_version_mismatch = session._ignore_version_mismatch
        run_parameters.reset_devices = session._reset_devices

        target_setup = convert_dsl_to_target_setup(session._device_setup)

        controller = ctrl.Controller(
            run_parameters=run_parameters,
            target_setup=target_setup,
            neartime_callbacks=session._neartime_callbacks,
        )
        controller.connect()
        session._controller = controller
        if session._connection_state.emulated:
            session._toolkit_devices = ctrl.MockedToolkit()
        else:
            session._toolkit_devices = ctrl.ToolkitDevices(controller._devices._devices)

    @staticmethod
    def disconnect(session: Session):
        controller: ctrl.Controller = session._controller
        controller.shut_down()
        controller.disconnect()
        session._controller = None
        session._toolkit_devices = ctrl.ToolkitDevices()

    @staticmethod
    def compile(
        session: Session, logger, compiler_settings: dict | None = None
    ) -> CompiledExperiment:
        logger.debug("Calling LabOne Q Compiler...")

        new_setup = convert_device_setup_to_setup(session.device_setup)
        new_experiment = convert_Experiment(session.experiment)
        signal_mapping = convert_signal_map(session.experiment)

        payload_builder = ApplicationManager.instance().payload_builder()
        payload = payload_builder.build_payload(
            new_setup,
            new_experiment,
            signal_mapping,
            compiler_settings,
        )

        compiled_experiment = CompiledExperiment(
            device_setup=session.device_setup,
            experiment=session.experiment,
            experiment_dict=None,  # deprecated
            scheduled_experiment=payload.scheduled_experiment,
        )
        return compiled_experiment

    @staticmethod
    def run(session: Session):
        controller: ctrl.Controller = session._controller

        if controller._run_parameters.shut_down is True:
            atexit.register(ctrl._stop_controller, controller)

        controller.execute_compiled_legacy(session.compiled_experiment, session)

    @staticmethod
    def replace_pulse(
        session: Session, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        controller: ctrl.Controller = session._controller
        controller.replace_pulse(pulse_uid, pulse_or_array)

    @staticmethod
    def init_logging(log_level=None, performance_log=None):
        import sys

        if "pytest" not in sys.modules:
            # Only initialize logging outside pytest
            # pytest initializes the logging itself
            ctrl.initialize_logging(
                log_level=log_level, performance_log=performance_log
            )
