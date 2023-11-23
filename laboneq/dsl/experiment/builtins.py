# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from functools import wraps
from typing import Any

import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType, AveragingMode, RepetitionMode
from laboneq.dsl import Parameter
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.experiment.experiment_context import (
    ExperimentContextManager,
    current_experiment_context,
)
from laboneq.dsl.experiment.pulse import Pulse

if typing.TYPE_CHECKING:
    from laboneq.dsl.experiment import Experiment, Section
    from laboneq.dsl.prng import PRNG

__all__ = [
    "acquire",
    "acquire_loop_rt",
    "add",
    "call",
    "case",
    "delay",
    "experiment_calibration",
    "experiment",
    "for_each",
    "map_signal",
    "match",
    "measure",
    "play",
    "qubit_experiment",
    "reserve",
    "section",
    "set_node",
    "sweep",
    "sweep_range",
]

from laboneq.dsl.experiment.section_context import (
    AcquireLoopNtSectionContextManager,
    AcquireLoopRtSectionContextManager,
    CaseSectionContextManager,
    MatchSectionContextManager,
    SectionContextManager,
    SweepSectionContextManager,
    active_section,
    PRNGSetupContextManager,
    PRNGLoopContextManager,
)


def _active_experiment() -> Experiment:
    context = current_experiment_context()
    if not context:
        raise LabOneQException("Not in an experiment context")
    return context.experiment


def section(*args, **kwargs):
    return SectionContextManager(*args, **kwargs)


def sweep(*args, parameter=None, **kwargs):
    parameters = parameter if isinstance(parameter, list) else [parameter]
    return SweepSectionContextManager(*args, parameters=parameters, **kwargs)


def acquire_loop_rt(
    count,
    averaging_mode=AveragingMode.CYCLIC,
    repetition_mode=RepetitionMode.FASTEST,
    repetition_time=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    uid=None,
    reset_oscillator_phase=False,
):
    return AcquireLoopRtSectionContextManager(
        count=count,
        averaging_mode=averaging_mode,
        repetition_mode=repetition_mode,
        repetition_time=repetition_time,
        acquisition_type=acquisition_type,
        uid=uid,
        reset_oscillator_phase=reset_oscillator_phase,
    )


def acquire_loop_nt(*args, **kwargs):
    return AcquireLoopNtSectionContextManager(*args, **kwargs)


def match(
    handle: str | None = None,
    user_register: int | None = None,
    uid: str | None = None,
    play_after: str | list[str] | None = None,
    local: bool | None = None,
):
    return MatchSectionContextManager(
        handle=handle,
        user_register=user_register,
        uid=uid,
        play_after=play_after,
        local=local,
    )


def case(state: int, uid: str | None = None):
    return CaseSectionContextManager(state=state, uid=uid)


def call(funcname, **kwargs):
    return active_section().call(funcname, **kwargs)


def play(
    signal,
    pulse,
    amplitude=None,
    phase=None,
    increment_oscillator_phase=None,
    set_oscillator_phase=None,
    length=None,
    pulse_parameters: dict[str, Any] | None = None,
    precompensation_clear: bool | None = None,
    marker=None,
):
    return active_section().play(
        signal=signal,
        pulse=pulse,
        amplitude=amplitude,
        phase=phase,
        increment_oscillator_phase=increment_oscillator_phase,
        set_oscillator_phase=set_oscillator_phase,
        length=length,
        pulse_parameters=pulse_parameters,
        precompensation_clear=precompensation_clear,
        marker=marker,
    )


def delay(
    signal: str,
    time: float | Parameter,
    precompensation_clear: bool | None = None,
):
    return active_section().delay(
        signal=signal, time=time, precompensation_clear=precompensation_clear
    )


def reserve(signal):
    return active_section().reserve(signal)


def acquire(
    signal: str,
    handle: str,
    kernel: Pulse | list[Pulse] | None = None,
    length: float | None = None,
    pulse_parameters: dict[str, Any] | list[dict[str, Any] | None] | None = None,
):
    return active_section().acquire(
        signal=signal,
        handle=handle,
        kernel=kernel,
        length=length,
        pulse_parameters=pulse_parameters,
    )


def measure(
    acquire_signal: str,
    handle: str,
    integration_kernel: Pulse | list[Pulse] | None = None,
    integration_kernel_parameters: dict[str, Any]
    | list[dict[str, Any] | None]
    | None = None,
    integration_length: float | None = None,
    measure_signal: str | None = None,
    measure_pulse: Pulse | None = None,
    measure_pulse_length: float | None = None,
    measure_pulse_parameters: dict[str, Any] | None = None,
    measure_pulse_amplitude: float | None = None,
    acquire_delay: float | None = None,
    reset_delay: float | None = None,
):
    return active_section().measure(
        acquire_signal=acquire_signal,
        handle=handle,
        integration_kernel=integration_kernel,
        integration_kernel_parameters=integration_kernel_parameters,
        integration_length=integration_length,
        measure_signal=measure_signal,
        measure_pulse=measure_pulse,
        measure_pulse_length=measure_pulse_length,
        measure_pulse_parameters=measure_pulse_parameters,
        measure_pulse_amplitude=measure_pulse_amplitude,
        acquire_delay=acquire_delay,
        reset_delay=reset_delay,
    )


def add(section: Section):
    try:
        parent = active_section()
    except LabOneQException:
        parent = _active_experiment()
    parent.add(section)


def set_node(path: str, value: Any):
    return active_section().set_node(path=path, value=value)


def sweep_range(start, stop, count, uid=None, axis_name=None, **kwargs):
    from laboneq.dsl import LinearSweepParameter

    param = LinearSweepParameter(
        start=start, stop=stop, count=count, axis_name=axis_name
    )
    return sweep(uid=uid or axis_name, parameter=param, **kwargs)


def for_each(iterable, uid=None, axis_name=None, **kwargs):
    from laboneq.dsl import SweepParameter

    param = SweepParameter(values=np.array(iterable), axis_name=axis_name)
    return sweep(uid=uid or axis_name, parameter=param, **kwargs)


def experiment(uid=None, signals=None):
    return ExperimentContextManager(uid=uid, signals=signals)


def qubit_experiment(qubits: list, **kwargs):
    def decorator(f):
        @wraps(f)
        def wrapper(*inner_args, **inner_kwargs):
            context = ExperimentContextManager(
                uid=f.__name__,
                signals=[s for q in qubits for s in q.experiment_signals()],
            )
            with context as experiment:
                f(*inner_args, **inner_kwargs)
            return experiment

        return wrapper

    return decorator


def experiment_calibration():
    """Get the calibration of the experiment in construction"""
    context = current_experiment_context()
    if context is None:
        raise LabOneQException("Not in an experiment context")
    context.calibration = context.calibration or Calibration()
    return context.calibration


def map_signal(experiment_signal_uid: str, logical_signal: LogicalSignal):
    _active_experiment().map_signal(experiment_signal_uid, logical_signal)


def prng_setup(range: int, seed=1, uid=None):
    from laboneq.dsl.prng import PRNG

    prng = PRNG(range, seed)
    return PRNGSetupContextManager(prng=prng, uid=uid)


def prng_loop(prng: PRNG, count=1, uid=None):
    from laboneq.dsl.prng import PRNGSample

    prng_sample = PRNGSample(uid=uid, prng=prng, count=count)
    return PRNGLoopContextManager(prng_sample, uid)
