# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import typing
from functools import wraps
from typing import Any, Iterable

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
from laboneq.dsl.experiment.uid_generator import (
    GLOBAL_UID_GENERATOR,
    reset_global_uid_generator,
)
import logging

if typing.TYPE_CHECKING:
    from laboneq.dsl.experiment import Experiment, Section, PlayPulse, Operation
    from laboneq.dsl.prng import PRNG, PRNGSample

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
    "play_indexed",
    "prng_setup",
    "prng_loop",
    "qubit_experiment",
    "reserve",
    "reset_global_uid_generator",
    "section",
    "set_node",
    "sweep",
    "sweep_range",
    "uid",
]

from laboneq.dsl.experiment.section_context import (
    AcquireLoopRtSectionContextManager,
    CaseSectionContextManager,
    MatchSectionContextManager,
    SectionContextManager,
    SweepSectionContextManager,
    active_section,
    PRNGSetupContextManager,
    PRNGLoopContextManager,
)


_logger = logging.getLogger(__name__)


def _active_experiment() -> Experiment:
    context = current_experiment_context()
    if not context:
        raise LabOneQException("Not in an experiment context")
    return context.experiment


def section(*args, **kwargs):
    return SectionContextManager(*args, **kwargs)


def sweep(*args, parameter=None, **kwargs):
    # TODO: Rework the interface of this function and context manager. Currently:
    #
    #       - positional arguments are accepted, but passing any positional argumnets
    #         gives obscure errors.
    #
    #       - one cannot easily get access to the created section
    return SweepSectionContextManager(*args, parameters=parameter, **kwargs)


def acquire_loop_rt(
    count,
    averaging_mode=AveragingMode.CYCLIC,
    repetition_mode=RepetitionMode.FASTEST,
    repetition_time=None,
    acquisition_type=AcquisitionType.INTEGRATION,
    uid=None,
    name=None,
    reset_oscillator_phase=False,
):
    return AcquireLoopRtSectionContextManager(
        count=count,
        averaging_mode=averaging_mode,
        repetition_mode=repetition_mode,
        repetition_time=repetition_time,
        acquisition_type=acquisition_type,
        uid=uid,
        name=name,
        reset_oscillator_phase=reset_oscillator_phase,
    )


def match(
    handle: str | None = None,
    user_register: int | None = None,
    prng_sample: PRNGSample | None = None,
    sweep_parameter: Parameter | None = None,
    uid: str | None = None,
    name: str | None = None,
    play_after: str | list[str] | None = None,
    local: bool | None = None,
):
    return MatchSectionContextManager(
        handle=handle,
        user_register=user_register,
        prng_sample=prng_sample,
        sweep_parameter=sweep_parameter,
        uid=uid,
        name=name,
        play_after=play_after,
        local=local,
    )


def case(state: int, uid: str | None = None, name: str | None = None):
    return CaseSectionContextManager(state=state, uid=uid, name=name)


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


def add(section: Section | Operation):
    try:
        parent = active_section()
    except LabOneQException:
        parent = _active_experiment()
    parent.add(section)


def set_node(path: str, value: Any):
    return active_section().set_node(path=path, value=value)


def sweep_range(start, stop, count, uid=None, axis_name=None, **kwargs):
    # TODO: Sort out the relationship between uid, axis_name and name.
    #       This function currently accepts all three, which definitely
    #       has potential to be confusing.
    from laboneq.dsl import LinearSweepParameter

    param = LinearSweepParameter(
        start=start, stop=stop, count=count, axis_name=axis_name
    )
    return sweep(uid=uid or axis_name, parameter=param, **kwargs)


def for_each(iterable, uid=None, axis_name=None, **kwargs):
    from laboneq.dsl import SweepParameter

    param = SweepParameter(values=np.array(iterable), axis_name=axis_name)
    return sweep(uid=uid or axis_name, parameter=param, **kwargs)


def experiment(uid=None, name=None, signals=None):
    return ExperimentContextManager(uid=uid, name=name, signals=signals)


def qubit_experiment(qubits: list, **kwargs):
    # TODO: rewrite this to allow qubits to be detected in the
    #       from arguments rather than passed to the decorator
    #       as they are here which results in always having to
    #       nest the decorated function inside another one, e.g.:
    #
    #       def rabi(q, *args, **kw):
    #
    #          @qubit_experiment([q])
    #          def actual_rabi(...):
    #              ...
    #
    #          return actual_rabi(*args, **kw)
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


def uid(prefix: str) -> str:
    """Generate a unique identifier.

    Generates an identifier that is unique to the experiment
    being constructed. The identifier is created by by appending a
    count to a given prefix.

    Arguments:
        prefix:
            The prefix for the unique identifier.

    Returns:
        A unique identifier.

    Raises:
        LabOneQException:
            If there is no experiment context.
    """
    context = current_experiment_context()
    if context is not None:
        return context.uid(prefix)
    return GLOBAL_UID_GENERATOR.uid(prefix)


def map_signal(experiment_signal_uid: str, logical_signal: LogicalSignal):
    _active_experiment().map_signal(experiment_signal_uid, logical_signal)


def prng_setup(range: int, seed=1, uid=None, name=None):
    from laboneq.dsl.prng import PRNG

    prng = PRNG(range, seed)
    return PRNGSetupContextManager(prng=prng, uid=uid, name=name)


def prng_loop(prng: PRNG, count=1, uid=None, name=None):
    from laboneq.dsl.prng import PRNGSample

    maybe_uid = {"uid": uid} if uid is not None else {}

    prng_sample = PRNGSample(prng=prng, count=count, **maybe_uid)
    return PRNGLoopContextManager(prng_sample, uid=uid, name=name)


def play_indexed(pulses: Iterable[PlayPulse], index: PRNGSample):
    # TODO: extend to allow passing all the other possibilities `match`
    #       accepts to match on and add tests to test_builtins.py.
    count = 0
    with match(prng_sample=index):
        for i, p in enumerate(pulses):
            count += 1
            with case(i):
                add(p)
    if count != index.prng.range:
        _logger.warning(
            f"'playIndexed' called with {count} pulses, mismatching the range of the PRNG, which is {index.prng.range}"
        )
