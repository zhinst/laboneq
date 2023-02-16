# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import atexit
from typing import TYPE_CHECKING, Dict

from numpy import typing as npt

from laboneq import controller as ctrl
from laboneq.compiler import Compiler
from laboneq.core.types import CompiledExperiment

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.session import Session


class LabOneQFacade:
    @staticmethod
    def connect(session: Session):
        run_parameters = ctrl.ControllerRunParameters()
        run_parameters.dry_run = session._connection_state.emulated
        run_parameters.ignore_lab_one_version_error = (
            session._ignore_lab_one_version_error
        )

        controller = ctrl.Controller(
            run_parameters=run_parameters,
            device_setup=session._device_setup,
            user_functions=session._user_functions,
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
        session: Session, logger, compiler_settings: Dict = None
    ) -> CompiledExperiment:
        logger.debug("Calling LabOne Q Compiler...")
        compiler = Compiler.from_user_settings(compiler_settings)
        compiled_experiment = compiler.run(
            {"setup": session.device_setup, "experiment": session.experiment}
        )
        compiled_experiment.device_setup = session.device_setup
        compiled_experiment.experiment = session.experiment
        return compiled_experiment

    @staticmethod
    def run(session: Session):
        controller: ctrl.Controller = session._controller

        if controller._run_parameters.shut_down is True:
            atexit.register(ctrl._stop_controller, controller)

        controller.execute_compiled(session.compiled_experiment, session)

    @staticmethod
    def replace_pulse(
        session: Session, pulse_uid: str | Pulse, pulse_or_array: npt.ArrayLike | Pulse
    ):
        controller: ctrl.Controller = session._controller
        controller.replace_pulse(pulse_uid, pulse_or_array)

    @staticmethod
    def init_logging(log_level=None, performance_log=None):
        import sys

        if not "pytest" in sys.modules:
            # Only initialize logging outside pytest
            # pytest initializes the logging itself
            ctrl.initialize_logging(
                log_level=log_level, performance_log=performance_log
            )
