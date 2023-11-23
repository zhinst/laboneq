# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from laboneq.application_management.application_manager import ApplicationManager
from laboneq.data.execution_payload import ExecutionPayload
from laboneq.data.experiment_description import (
    Acquire,
    AcquireLoopRt,
    ExecutionType,
    Experiment,
    PlayPulse,
    Section,
    Sweep,
)
from laboneq.data.experiment_results import ExperimentResults
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.data.setup_description import Setup
from laboneq.data.setup_description.setup_helper import SetupHelper
from laboneq.interfaces.experiment.experiment_api import ExperimentAPI


class OscillatorGetter:
    def __init__(self, logical_signal):
        self.logical_signal = logical_signal

    def __setattr__(self, __name: str, __value):
        if hasattr(self, "logical_signal"):
            if hasattr(self.logical_signal, "calibration"):
                if hasattr(self.logical_signal.calibration.oscillator, __name):
                    self.logical_signal.calibration.oscillator.__setattr__(
                        __name, __value
                    )
                    return
        self.__dict__[__name] = __value


class ExperimentResultsAdapter:
    def __init__(self, experiment_results):
        self.experiment_results = experiment_results
        self.compiled_experiment = ScheduledExperiment()

    def __getattr__(self, __name: str):
        if hasattr(self.experiment_results, __name):
            return self.experiment_results.__getattribute__(__name)
        else:
            if __name == "device_setup":
                return None
            if __name == "get_data":
                return self.get_data
            if __name == "get_axis":
                return self.get_axis

    def get_data(self, *args, **kwargs):
        return self.compiled_experiment

    def get_axis(self, *args, **kwargs):
        return [0]

    def get_axis_name(self, *args, **kwargs):
        return ["dummy"]


class DeviceSetupAdapter:
    def from_descriptor(
        yaml_text: str,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ):
        from laboneq.application_management.application_manager import (
            ApplicationManager,
        )

        l1q: ExperimentAPI = ApplicationManager.instance().laboneq()
        retval = l1q.device_setup_from_descriptor(
            yaml_text, server_host, server_port, setup_name
        )
        for ls in SetupHelper(retval).logical_signals():
            ls.oscillator = OscillatorGetter(ls)
        return retval


class SignalCalibrationAdapter:
    def __init__(self, *args, **kwargs):
        self.oscillator = None


class LegacySessionAdapter:
    def __init__(self, device_setup: Setup):
        self.compiled_experiment = ScheduledExperiment()
        self.results = ExperimentResultsAdapter(ExperimentResults())
        app_manager = ApplicationManager.instance()
        self.l1q = app_manager.laboneq()
        self.l1q.set_current_setup(device_setup)
        self.do_emulation = False
        self.reset_devices = False

    def connect(
        self,
        do_emulation: bool = False,
        ignore_version_mismatch=False,
        reset_devices=False,
    ):
        self.do_emulation = do_emulation
        self.reset_devices = reset_devices

    def compile(self, experiment: Experiment) -> ExecutionPayload:
        self.l1q.set_current_experiment(experiment.data_experiment)
        self.l1q.map_signals(experiment.signal_mappings)
        self.compiled_experiment = self.l1q.build_payload_for_current_experiment()
        return self.compiled_experiment

    def run(self, compiled_experiment: ExecutionPayload | None = None):
        if compiled_experiment is None:
            compiled_experiment = self.compiled_experiment
        self.results = self.l1q.run_payload(compiled_experiment)
        return self.results

    def get_results(self, *args, **kwargs):
        return self.results


class ExperimentAdapter:
    def __init__(self, uid=None, signals=None):
        self.uid = uid
        if signals is None:
            signals = []

        self.signal_mappings = {}
        self.data_experiment = Experiment()
        self.data_experiment.signals = signals
        self._section_stack = []

    def map_signal(self, experiment_signal_uid: str, logical_signal):
        if experiment_signal_uid not in {s.uid for s in self.data_experiment.signals}:
            raise ValueError(
                "Signal {} not found in experiment".format(experiment_signal_uid)
            )
        self.signal_mappings[experiment_signal_uid] = (
            logical_signal.group + "/" + logical_signal.name
        )

    def sweep(self, uid=None, parameter=None):
        section = Sweep(uid=uid, parameters=[parameter])
        return SectionContext(self, section)

    def acquire_loop_rt(
        self,
        uid=None,
        acquisition_type=None,
        averaging_mode=None,
        count=None,
        repetition_mode=None,
        repetition_time=None,
        reset_oscillator_phase=False,
    ):
        section = AcquireLoopRt(
            uid=uid,
            acquisition_type=acquisition_type,
            averaging_mode=averaging_mode,
            count=count,
            execution_type=ExecutionType.REAL_TIME,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
            reset_oscillator_phase=reset_oscillator_phase,
        )
        return SectionContext(self, section)

    def section(self, uid=None, execution_type=None):
        section = Section(uid=uid, execution_type=execution_type)
        return SectionContext(self, section)

    def play(
        self,
        signal,
        pulse,
        amplitude=None,
        phase=None,
        increment_oscillator_phase=None,
        set_oscillator_phase=None,
        length=None,
        pulse_parameters=None,
        precompensation_clear=None,
        marker=None,
    ):
        self._register_pulse(pulse)
        operation = PlayPulse(
            signal=signal,
            pulse=pulse,
            amplitude=amplitude,
            increment_oscillator_phase=increment_oscillator_phase,
            phase=phase,
            set_oscillator_phase=set_oscillator_phase,
            length=length,
            pulse_parameters=pulse_parameters,
            precompensation_clear=precompensation_clear,
            marker=marker,
        )
        self._push_operation(operation)

    def acquire(self, signal, handle, length):
        operation = Acquire(signal=signal, handle=handle, length=length)
        self._push_operation(operation)

    def _push_section(self, section):
        if section.execution_type is None:
            if self._section_stack:
                parent_section = self._peek_section()
                execution_type = parent_section.execution_type
            else:
                execution_type = ExecutionType.NEAR_TIME
            section.execution_type = execution_type
        self._section_stack.append(section)

    def _push_operation(self, operation):
        section = self._peek_section()
        section.children.append(operation)

    def _pop_and_add_section(self):
        if not self._section_stack:
            raise ValueError(
                "Internal error: Section stack should not be empty. Unbalanced push/pop."
            )
        section = self._section_stack.pop()
        self._add_section_to_current_section(section)

    def _add_section_to_current_section(self, section):
        if not self._section_stack:
            self.data_experiment.sections.append(section)
        else:
            current_section = self._section_stack[-1]
            current_section.children.append(section)

    def _peek_section(self):
        if not self._section_stack:
            raise ValueError(
                "No section in experiment. Use 'with your_exp.section(...):' to create a section scope first."
            )
        return self._section_stack[-1]

    def _peek_rt_section(self):
        if not self._section_stack:
            raise ValueError(
                "No section in experiment. Use 'with your_exp.section(...):' to create a section scope first."
            )
        for s in reversed(self._section_stack):
            if s.execution_type == ExecutionType.REAL_TIME:
                return s
        raise ValueError(
            "No surrounding realtime section in experiment. Use 'with your_exp.acquire_loop_rt(...):' to create a section scope first."
        )

    def _register_pulse(self, pulse):
        if pulse.uid is None:
            pulse.uid = "pulse_{}".format(len(self.data_experiment.pulses))
        if pulse.uid in {p.uid for p in self.data_experiment.pulses}:
            return
        self.data_experiment.pulses.append(pulse)
        return pulse


class SectionContext:
    def __init__(self, experiment, section):
        self.exp = experiment
        self.section = section

    def __enter__(self):
        self.exp._push_section(self.section)
        return self.section

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exp._pop_and_add_section()
