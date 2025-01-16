# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import typing
from typing import Any, Iterable

import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import (
    AcquisitionType,
    AveragingMode,
    RepetitionMode,
    SectionAlignment,
)
from laboneq.dsl import Parameter
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.experiment.experiment_context import (
    ExperimentContextManager,
    current_experiment_context,
)
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.section_context import (
    AcquireLoopRtSectionContextManager,
    CaseSectionContextManager,
    MatchSectionContextManager,
    SectionContextManager,
    SweepSectionContextManager,
    current_section_context,
    PRNGSetupContextManager,
    PRNGLoopContextManager,
)
from laboneq.dsl.experiment.uid_generator import (
    GLOBAL_UID_GENERATOR,
    reset_global_uid_generator,
)

if typing.TYPE_CHECKING:
    from typing import Callable

    from laboneq.dsl.enums import ExecutionType
    from laboneq.dsl.experiment import (
        Experiment,
        Section,
        PlayPulse,
        Operation,
        ExperimentSignal,
    )
    from laboneq.dsl.prng import PRNG, PRNGSample

__all__ = [
    "acquire",
    "acquire_loop_rt",
    "active_section",
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
    "reserve",
    "reset_global_uid_generator",
    "section",
    "set_node",
    "sweep",
    "sweep_range",
    "uid",
]

_logger = logging.getLogger(__name__)


def _active_experiment() -> Experiment:
    context = current_experiment_context()
    if not context:
        raise LabOneQException("Not in an experiment context")
    return context.experiment


def active_section() -> Section:
    """Return the currently active section.

    Returns:
        The currently active section.

    Raises:
        LabOneQException:
            If no section is active.
    """
    context = current_section_context()
    if not context:
        raise LabOneQException("Must be in a section context")
    return context.section


def section(
    *,
    uid: str | None = None,
    name: str = "unnamed",
    alignment: SectionAlignment = SectionAlignment.LEFT,
    execution_type: ExecutionType | None = None,
    length: float | None = None,
    play_after: str | Section | list[str | Section] | None = None,
    trigger: dict[str, dict] | None = None,
    on_system_grid: bool = False,
) -> SectionContextManager:
    """Returns a new section context manager.

    The arguments are the same as those of [Section][laboneq.dsl.experiment.section.Section], except that
    `children` may not be directly supplied.

    Upon entering the returned context manager a [Section][laboneq.dsl.experiment.section.Section]
    is returned. Within the context block, operations and subsections are automatically added to the
    section.

    The section context manager may also be used as a decorator. The function being decorated will be
    executed within the section context and the resulting section will be returned.

    Arguments:
        uid:
            Unique identifier for the section. Maybe be omitted if one
            is not required.
        name:
            A name for the section. The name need not be unique.
            The name is used as a prefix for generating a `uid` for the section if one is not specified.
        alignment:
            Specifies the time alignment of operations and sections within
            this section. Left alignment positions operations and sections
            as early in time as possible, right alignment positions them
            as late as possible.
        execution_type:
            Whether the section is near-time or real-time.
            If `None` the section infers its execution type from its
            parent.
        length:
            Minimum length of the section in seconds. The scheduled section
            length will be the greater of this minimum length and the
            length of the section contents, plus a small extra amount of
            times so that the section length is a multiple of the section
            timing grid.
            If `None`, the section has no minimum length and will be as
            short as possible.
        play_after:
            A list of sections that must complete before this section
            may be played.
            If `None`, the section is played as soon as allowed by the
            signal lines required.
        trigger:
            Optional trigger pulses to play during this section.
            See [Experiment.section][laboneq.dsl.experiment.experiment.Experiment.section].
        on_system_grid:
            If True, the section boundaries are always rounded to the system grid,
            even if the contained signals would allow for tighter alignment.

    Returns:
        A section context manager.
    """
    return SectionContextManager(
        uid=uid,
        name=name,
        alignment=alignment,
        execution_type=execution_type,
        length=length,
        play_after=play_after,
        trigger=trigger,
        on_system_grid=on_system_grid,
    )


def sweep(
    parameter: list[Parameter] | Parameter,
    *,
    chunk_count: int = 1,
    reset_oscillator_phase: bool = False,
    uid: str | None = None,
    name: str = "unnamed",
    alignment: SectionAlignment = SectionAlignment.LEFT,
) -> SweepSectionContextManager:
    """Returns a sweep context manager.

    The arguments are the same as those of [Sweep][laboneq.dsl.experiment.section.Sweep], except that
    `children`, `length`, `play_after`, `trigger` and `on_system_grid` may not be directly supplied.

    Upon entering the returned context manager a parameter is returned if only a single parameter is
    being swept. Otherwise a tuple of swepted parameters is returned.

    Within the context block, operations and subsections are automatically added to the
    sweep section and the active section may be retrieved by calling `active_section(...)`.

    Arguments:
        parameter:
            Parameters that should be swept.
        chunk_count (int):
            Split the sweep into N chunks.
            Default: `1`.
        reset_oscillator_phase (bool):
            When True, reset all oscillators at the start of every step.
            Default: `False`.
        uid:
            Unique identifier for the section. Maybe be omitted if one
            is not required.
        name:
            A name for the section. The name need not be unique.
            The name is used as a prefix for generating a `uid` for the section if one is not specified.
        alignment:
            Specifies the time alignment of operations and sections within
            this section. Left alignment positions operations and sections
            as early in time as possible, right alignment positions them
            as late as possible.

    Returns:
        A sweep section context manager.
    """
    return SweepSectionContextManager(
        uid=uid,
        name=name,
        parameters=parameter,
        alignment=alignment,
        reset_oscillator_phase=reset_oscillator_phase,
        chunk_count=chunk_count,
    )


def acquire_loop_rt(
    count: int,
    *,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
    averaging_mode: AveragingMode = AveragingMode.CYCLIC,
    repetition_mode: RepetitionMode = RepetitionMode.FASTEST,
    repetition_time: float | None = None,
    reset_oscillator_phase: bool = False,
    uid: str | None = None,
    name: str = "unnamed",
) -> AcquireLoopRtSectionContextManager:
    """Returns an acquire loop section context manager.

    The arguments are the same as those of [AcquireLoopRt][laboneq.dsl.experiment.section.AcquireLoopRt].

    Upon entering the returned context manager an [AcquireLoopRt][laboneq.dsl.experiment.section.AcquireLoopRt]
    section is returned. Within the context block, operations and subsections are automatically added to the
    section.

    Arguments:
        count:
            Number of loops to perform.
        acquisition_type:
            Type of the acquisition. One of integration, spectroscopy,
            discrimination or RAW.
        averaging_mode:
            Averaging method. One of sequential, cyclic or single shot.
        repetition_mode:
            Repetition method. One of fastest, constant or auto.
        repetition_time:
            The repetition time, when `repetition_mode` is
            [RepetitionMode.CONSTANT][laboneq.core.types.enums.repetition_mode.RepetitionMode].
        reset_oscillator_phase:
            When true, reset all oscillators at the start of every step.
        uid:
            Unique identifier for the section. Maybe be omitted if one
            is not required.
        name:
            A name for the section. The name need not be unique.
            The name is used as a prefix for generating a `uid` for the section if one is not specified.

    Returns:
        An real-time acquire loop context manager.
    """
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
    *,
    user_register: int | None = None,
    prng_sample: PRNGSample | None = None,
    sweep_parameter: Parameter | None = None,
    local: bool | None = None,
    uid: str | None = None,
    name: str | None = None,
    play_after: str | list[str] | None = None,
) -> MatchSectionContextManager:
    """Returns a match section context manager.

    The arguments are the same as those of [Match][laboneq.dsl.experiment.section.Match], except that
    `children` may not be directly supplied.

    Upon entering the returned context manager a [Match][laboneq.dsl.experiment.section.Match]
    is returned. Within the context block, operations and subsections are automatically added to the
    match section.

    Only `case(...)` sections may appear as children of a match section.

    Arguments:
        handle:
            Handle from which to obtain results.
            See [Section.measure][laboneq.dsl.experiment.section.Section.measure]
            and [Section.acquire][laboneq.dsl.experiment.section.Section.acquire]
            for where handles are specified.
        user_register:
            User register on which to match.
        prng_sample:
            PRNG sample to match.
        sweep_parameter:
            Sweep parameter to match.
        local:
            Whether to fetch the codeword via the PQSC (`False`),
            SHFQC-internal bus (`True`) or automatic (`None`).
        uid:
            Unique identifier for the section. Maybe be omitted if one
            is not required.
        name:
            A name for the section. The name need not be unique.
            The name is used as a prefix for generating a `uid` for the section if one is not specified.
        play_after:
            A list of sections that must complete before this section
            may be played.
            If `None`, the section is played as soon as allowed by the
            signal lines required.
    """
    return MatchSectionContextManager(
        uid=uid,
        name=name,
        handle=handle,
        user_register=user_register,
        prng_sample=prng_sample,
        sweep_parameter=sweep_parameter,
        local=local,
        play_after=play_after,
    )


def case(
    state: int,
    *,
    uid: str | None = None,
    name: str = "unnamed",
) -> CaseSectionContextManager:
    """Returns a case section context manager.

    Upon entering the returned context manager a [Case][laboneq.dsl.experiment.section.Case]
    is returned. Within the context block, operations and subsections are automatically added to the
    match section.

    Case sections may appear as children of a match section.

    Arguments:
        state:
            Which state value this case is for.
        uid:
            Unique identifier for the section. Maybe be omitted if one
            is not required.
        name:
            A name for the section. The name need not be unique.
            The name is used as a prefix for generating a `uid` for the section if one is not specified.

    Raises:
        LabOneQException:
            Upon entering the context manager if the case section would be created outside of a
            match section.
    """
    return CaseSectionContextManager(state=state, uid=uid, name=name)


def call(funcname: str | Callable, **kwargs: object):
    """Call a near-time function in the active section.

    Adds a call operation to the active section.

    Arguments:
        funcname:
            Function that should be called.
        kwargs:
            Arguments of the function call.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().call(funcname, **kwargs)


def play(
    signal: str,
    pulse: Pulse,
    amplitude: float | complex | Parameter | None = None,
    phase: float | None = None,
    increment_oscillator_phase: float | Parameter | None = None,
    set_oscillator_phase: float | None = None,
    length: float | Parameter | None = None,
    pulse_parameters: dict[str, Any] | None = None,
    precompensation_clear: bool | None = None,
    marker: dict | None = None,
):
    """Play a pulse on the given signal in the active section.

    Adds a play pulse operation to the active section.

    Arguments:
        signal:
            The name of the signal to play the pulse on.
        pulse:
            Pulse that should be played on the signal.
        amplitude:
            Amplitude of the pulse that should be played.
        phase:
            Phase of the pulse that should be played.
        increment_oscillator_phase:
            Increment the phase angle of the modulating oscillator at the start of
            playing this pulse by this angle (in radians).
        set_oscillator_phase:
            Set the phase of the modulating oscillator at the start of playing this
            pulse to this angle (in radians).
        length:
            Modify the length of the pulse to the given value.
        pulse_parameters:
            Dictionary with user pulse function parameters (re)binding.
        precompensation_clear:
            Clear the precompensation filter during the pulse.
        marker:
            Instruction for playing marker signals along with the pulse.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().play(
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
    *,
    precompensation_clear: bool | None = None,
):
    """Apply a delay on the given signal in the active section.

    Adds a delay operation to the active section.

    Arguments:
        signal:
            The name of the signal to delay on.
        time:
            The duration of the delay (in seconds).
        precompensation_clear:
            If true, clear the precompensation filter of the signal.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().delay(
        signal=signal,
        time=time,
        precompensation_clear=precompensation_clear,
    )


def reserve(signal: str):
    """Reserve the given signal in the active section.

    Adds a reserve operation to the active section.

    Arguments:
        signal:
            The name of the signal to reserve.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().reserve(signal)


def acquire(
    signal: str,
    handle: str,
    kernel: Pulse | list[Pulse] | None = None,
    length: float | None = None,
    pulse_parameters: dict[str, Any] | list[dict[str, Any] | None] | None = None,
):
    """Perform an acquisition on the given signal in the active section.

    Adds an acquire operation to the active section.

    Arguments:
        signal:
            A string that specifies the signal for the acquisition.
        handle:
            A string that specifies the handle of the acquired results.
        kernel:
            An optional Pulse object that specifies the kernel for integration.
            In case of multistate discrimination, a list of kernels.
        length:
            An optional float that specifies the integration length.
        pulse_parameters:
            An optional dictionary that contains pulse parameters for the integration kernel.
            In case of multistate discrimination, a list of dictionaries.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().acquire(
        signal=signal,
        handle=handle,
        kernel=kernel,
        length=length,
        pulse_parameters=pulse_parameters,
    )


def measure(
    *,
    acquire_signal: str,
    handle: str,
    integration_kernel: Pulse | list[Pulse] | None = None,
    integration_kernel_parameters: (
        dict[str, Any] | list[dict[str, Any] | None] | None
    ) = None,
    integration_length: float | None = None,
    measure_signal: str | None = None,
    measure_pulse: Pulse | None = None,
    measure_pulse_length: float | None = None,
    measure_pulse_parameters: dict[str, Any] | None = None,
    measure_pulse_amplitude: float | None = None,
    acquire_delay: float | None = None,
    reset_delay: float | None = None,
):
    """Add a measurement to the active section.

    A measurement consists of an optional playback of a measurement pulse,
    the acquisition of the return signal and an optional delay after the
    end of the acquisition.

    Both spectroscopy and ordinary measurements are supported:

    - For pulsed spectroscopy, set `integration_length` and either `measure_pulse` or `measure_pulse_length`.
    - For CW spectroscopy, set only `integration_length` and do not specify the measure signal.
    - For all other measurements, set either length or pulse for both the measure pulse and integration kernel.

    Arguments:
        acquire_signal:
            A string that specifies the signal for the data acquisition.
        handle:
            A string that specifies the handle of the acquired results.
        integration_kernel:
            An optional Pulse object that specifies the kernel for integration.
            In case of multistate discrimination, a list of kernels.
        integration_kernel_parameters:
            An optional dictionary that contains pulse parameters for the integration kernel.
            In case of multistate discrimination, a list of dictionaries.
        integration_length:
            An optional float that specifies the integration length.
        measure_signal:
            An optional string that specifies the signal to measure.
        measure_pulse:
            An optional Pulse object that specifies the readout pulse for measurement.

            If this parameter is not supplied, no pulse will be played back for the measurement,
            which enables CW spectroscopy on SHFQA instruments.
        measure_pulse_length:
            An optional float that specifies the length of the measurement pulse.
        measure_pulse_parameters:
            An optional dictionary that contains parameters for the measurement pulse.
        measure_pulse_amplitude:
            An optional float that specifies the amplitude of the measurement pulse.
        acquire_delay:
            An optional float that specifies the delay between the acquisition and the measurement.
        reset_delay:
            An optional float that specifies the delay after the acquisition to allow for state
            relaxation or signal processing.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().measure(
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
    """Add the given section or operation to the active section or experiment.

    Arguments:
        section:
            The section or operation to add.

    Raises:
        LabOneQException:
            If there is no active section and no active experiment.
    """
    try:
        parent = active_section()
    except LabOneQException:
        parent = _active_experiment()
    parent.add(section)


def set_node(path: str, value: Any):
    """Set the value of an instrument node.

    Adds a set node operation to the active section.

    Arguments:
        path:
            Path to the node whose value should be set.
        value:
            Value that should be set.

    Raises:
        LabOneQException:
            If there is no active section.
    """
    active_section().set_node(path=path, value=value)


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


def experiment(
    *,
    uid: str | None = None,
    name: str | None = None,
    signals: dict[str, ExperimentSignal] | list[ExperimentSignal | str] = None,
) -> ExperimentContextManager:
    """Returns an experiment context manager.

    The arguments are the same as those of [Experiment][laboneq.dsl.experiment.experiment.Experiment],
    except that `sections`, `version` and `epsilon` may not be directly supplied.

    Upon entering the returned context manager an [Experiment][laboneq.dsl.experiment.experiment.Experiment]
    is returned. Within the context block, operations and subsections are automatically added to the
    experiment.

    The experiment context manager may also be used as a decorator. The function being decorated will be
    executed withint the experiment context and the resulting experiment will be returned.

    Arguments:
        uid:
            Unique identifier for the experiment.
            If not specified, it will default to `None` unless the experiment context is used as a
            decorator, in which case it will default to the name of the decorated function.
        name:
            A name for the experiment. The name need not be unique.
            If not specified, it will default to "unnamed" unless the experiment context is used as a
            decorator, in which case it will default to the name of the decorated function.
        signals:
            Experiment signals, if any.

    Returns:
        A new experiment context manager.
    """
    return ExperimentContextManager(uid=uid, name=name, signals=signals)


def experiment_calibration() -> Calibration:
    """Return the calibration of the active experiment in context.

    Returns:
        The experiment calibration of the active experiment.

    Raises:
        LabOneQException:
            If there is no active experiment context.
    """
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
            If there is no active experiment context.
    """
    context = current_experiment_context()
    if context is not None:
        return context.uid(prefix)
    return GLOBAL_UID_GENERATOR.uid(prefix)


def map_signal(experiment_signal: str, logical_signal: LogicalSignal | str):
    """Connect an experiment signal to a logical signal.

    This connects a signal on the active experiment to a logical signal
    on a device.

    Arguments:
        experiment_signal:
            The name of experiment signal.
        logical_signal:
            The logical signal to connect to the experiment signal.

    !!! note
        One can also call [Experiment.map_signal][laboneq.dsl.experiment.Experiment.map_signal]
        directly on the experiment after it is created.
    """
    _active_experiment().map_signal(experiment_signal, logical_signal)


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
