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
        if not session._connection_state.emulated:
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
        session: Session, logger, do_simulation=False, compiler_settings: Dict = None
    ) -> CompiledExperiment:
        logger.debug("Calling LabOne Q Compiler...")
        if compiler_settings is not None:
            compiler_settings = {
                k: v
                for k, v in compiler_settings.items()
                if k
                in (
                    "MAX_EVENTS_TO_PUBLISH",
                    "PHASE_RESOLUTION_BITS",
                    "HDAWG_MIN_PLAYWAVE_HINT",
                    "HDAWG_MIN_PLAYZERO_HINT",
                    "UHFQA_MIN_PLAYWAVE_HINT",
                    "UHFQA_MIN_PLAYZERO_HINT",
                    "SHFQA_MIN_PLAYWAVE_HINT",
                    "SHFQA_MIN_PLAYZERO_HINT",
                    "SHFSG_MIN_PLAYWAVE_HINT",
                    "SHFSG_MIN_PLAYZERO_HINT",
                    "EMIT_TIMING_COMMENTS",
                    "HDAWG_FORCE_COMMAND_TABLE",
                    "SHFSG_FORCE_COMMAND_TABLE",
                )
            }
        compiler = Compiler(compiler_settings)
        compiled_experiment = compiler.run(
            {"setup": session.device_setup, "experiment": session.experiment}
        )
        compiled_experiment.device_setup = session.device_setup
        compiled_experiment.experiment = session.experiment
        if do_simulation:
            compiled_experiment.output_signals = LabOneQFacade.simulate_outputs(
                compiled_experiment, session.max_simulation_time, logger
            )
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
    def simulate_outputs(
        compiled_experiment: CompiledExperiment, max_simulation_time: float, logger
    ):
        from laboneq.core.types.device_output_signals import DeviceOutputSignals
        from laboneq.dsl.result import Waveform
        from laboneq.simulator import analyze_compiler_output_memory

        try:

            output_signals = DeviceOutputSignals()
            simulated_outputs = analyze_compiler_output_memory(
                compiled_experiment, max_simulation_time
            )

            for simulated_result in simulated_outputs.values():
                device_uid = simulated_result.device_uid

                awg_index = simulated_result.awg_index
                channel_index = 0
                for k, v in sorted(simulated_result.output.items()):

                    time_axis = simulated_result.times[k]
                    time_axis_at_port = simulated_result.times_at_port[k]
                    waveform = Waveform(
                        uid=str(k),
                        data=v,
                        sampling_frequency=simulated_result.sample_frequency,
                        time_axis=time_axis,
                        time_axis_at_port=time_axis_at_port,
                    )
                    output_signals.map(
                        device_uid=device_uid,
                        signal_uid=str(awg_index),
                        channel_index=channel_index,
                        waveform=waveform,
                    )
                    channel_index += 1
        except Exception as e:
            logger.warning("Experiment simulation failed: %s", e)
            raise e

        return output_signals

    @staticmethod
    def init_logging(log_level=None, performance_log=None):
        import sys

        if not "pytest" in sys.modules:
            # Only initialize logging outside pytest
            # pytest initializes the logging itself
            ctrl.initialize_logging(
                log_level=log_level, performance_log=performance_log
            )
