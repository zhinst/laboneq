# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
import warnings
from typing import TYPE_CHECKING, Any

from laboneq._utils import id_generator
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import SectionAlignment
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.core.validators import validating_allowed_values
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    ExecutionType,
    RepetitionMode,
)
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.parameter import Parameter

from .acquire import Acquire
from .call import Call
from .delay import Delay
from .operation import Operation
from .play_pulse import PlayPulse
from .reserve import Reserve
from .set_node import SetNode

if TYPE_CHECKING:
    from ..prng import PRNG, PRNGSample


@classformatter
@attrs.define(slots=False)
class Section:
    """Representation of a section. A section is a logical concept that groups multiple operations into a single entity
    that can be though of a container. A section can either contain other sections or a list of operations (but not both
    at the same time). Operations within a section can be aligned in various ways (left, right). Sections can have a offset
    and/or a predefined length, and they can be specified to play after another section.

    Attributes:
        uid (str | None):
            Unique identifier for the section. Maybe be omitted if one
            is not required. Default: `None`.
        name (str):
            A name for the section. The name need not be unique.
            The name may, in future, be used as a prefix for
            generating a `uid` for the section if one is not specified.
            Default: `"unnamed"`.
        alignment (SectionAlignment):
            Specifies the time alignment of operations and sections within
            this section. Left alignment positions operations and sections
            as early in time as possible, right alignment positions them
            as late as possible.
            Default: [SectionAlignment.LEFT][laboneq.core.types.enums.section_alignment.SectionAlignment].
        execution_type (ExecutionType | None):
            Whether the section is near-time or real-time.
            If `None` the section infers its execution type from its
            parent.
            Default: `None`.
        length (float | None):
            Minimum length of the section in seconds. The scheduled section
            length will be the greater of this minimum length and the
            length of the section contents, plus a small extra amount of
            times so that the section length is a multiple of the section
            timing grid.
            If `None`, the section has no minimum length and will be as
            short as possible.
            Default: `None`.
        play_after (str | Section | list[str | Section] | None):
            A list of sections that must complete before this section
            may be played.
            If `None`, the section is played as soon as allowed by the
            signal lines required.
            Default: `None`.
        children (list[Section | Operation]):
            List of children. Each child may be another section or an
            operation.
            Default: `[]`.
        trigger (dict[str, dict[str, int]]):
            Optional trigger pulses to play during this section.
            See [Experiment.section][laboneq.dsl.experiment.experiment.Experiment.section].
            Default: `{}`.
        on_system_grid (bool):
            If True, the section boundaries are always rounded to the system grid,
            even if the contained signals would allow for tighter alignment.
            Default: `False`.

    !!! version-changed "Changed in version 2.0.0"
        Removed `offset` member variable.

    !!! version-changed "Added in version 2.26.0"
        Added `name` member variable.
    """

    # Unique identifier of the section.
    uid: str | None = attrs.field(default=None)

    # Non-unique name for the section.
    name: str = attrs.field(default="unnamed")

    # Alignment of operations and subsections within this section.
    alignment: SectionAlignment = attrs.field(default=SectionAlignment.LEFT)

    execution_type: ExecutionType | None = attrs.field(default=None)

    # Minimal length of the section in seconds. The scheduled section might be slightly longer, as its length is rounded to the next multiple of the section timing grid.
    length: float | None = attrs.field(default=None)

    # Play after the section with the given ID.
    play_after: str | Section | list[str | Section] | None = attrs.field(default=None)

    # List of children. Each child may be another section or an operation.
    children: list[Section | Operation] = attrs.field(factory=list)

    # Optional trigger pulses to play during this section.
    # See [Experiment.section][laboneq.dsl.experiment.experiment.Experiment.section].
    trigger: dict[str, dict[str, int]] = attrs.field(factory=dict)

    # Whether to escalate to the system grid even if tighter alignment is possible.
    # See [Experiment.section][laboneq.dsl.experiment.experiment.Experiment.section].
    on_system_grid: bool | None = attrs.field(default=False)

    def __attrs_post_init__(self):
        if self.uid is None:
            self.uid = id_generator("s")

    def add(self, section: Section | Operation | SetNode):
        """Add a subsection or operation to the section.

        Arguments:
            section: Item that is added.
        """
        self.children.append(section)

    @property
    def sections(self) -> tuple[Section, ...]:
        """A list of subsections of this section."""
        return tuple([s for s in self.children if isinstance(s, Section)])

    @property
    def operations(self) -> tuple[Operation, ...]:
        """A list of operations in the section.

        Note that there may be other children of a section which are not operations but subsections.
        """
        return tuple([s for s in self.children if isinstance(s, Operation)])

    def set_node(self, path: str, value: Any):
        """Set the value of an instrument node.

        Arguments:
            path: Path to the node whose value should be set.
            value: Value that should be set.
        """
        self.add(SetNode(path=path, value=value))

    def play(
        self,
        signal: str,
        pulse: Pulse,
        amplitude: float | complex | Parameter | None = None,
        phase: float | None = None,
        increment_oscillator_phase: float | Parameter | None = None,
        set_oscillator_phase: float | None = None,
        length: float | Parameter | None = None,
        pulse_parameters: dict[str, Any] | None = None,
        precompensation_clear: bool | None = None,
        marker: dict[str, Any] | None = None,
    ):
        """Play a pulse on a signal.

        Arguments:
            signal:
                Signal the pulse should be played on.
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
        """
        self.add(
            PlayPulse(
                signal,
                pulse,
                amplitude=amplitude,
                phase=phase,
                increment_oscillator_phase=increment_oscillator_phase,
                set_oscillator_phase=set_oscillator_phase,
                length=length,
                pulse_parameters=pulse_parameters,
                precompensation_clear=precompensation_clear,
                marker=marker,
            )
        )

    def reserve(self, signal: str):
        """Operation to reserve a signal for the active section.

        Reserving an experiment signal in a section means that if there is no
        osperation defined on that signal, it is not available for other sections
        as long as the active section is scoped.

        Arguments:
            signal: Signal that should be reserved.
        """
        self.add(Reserve(signal))

    def acquire(
        self,
        signal: str,
        handle: str,
        kernel: Pulse | list[Pulse] | None = None,
        length: float | None = None,
        pulse_parameters: dict[str, Any] | list[dict[str, Any] | None] | None = None,
    ):
        """Acquisition of results of a signal.

        Arguments:
            signal: Unique identifier of the signal where the result should be acquired.
            handle: Unique identifier of the handle that will be used to access the acquired result.
            kernel: Pulse base used for the acquisition. In case of multistate discrimination, a list of kernels.
            length: Integration length (only valid in spectroscopy mode).
            pulse_parameters: Dictionary with user pulse function parameters (re)binding. In case of multistate discrimination, a list of dicts.
        """
        self.add(
            Acquire(
                signal=signal,
                handle=handle,
                kernel=kernel,
                length=length,
                pulse_parameters=pulse_parameters,
            )
        )

    def measure(
        self,
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
        """
        Execute a measurement.

        Unifies the optional playback of a measurement pulse, the acquisition of the return signal and an optional delay after the signal acquisition.

        For pulsed spectroscopy, set `integration_length` and either `measure_pulse` or `measure_pulse_length`.
        For CW spectroscopy, set only `integration_length` and do not specify the measure signal.
        For all other measurements, set either length or pulse for both the measure pulse and integration kernel.

        Arguments:
            acquire_signal: A string that specifies the signal for the data acquisition.
            handle: A string that specifies the handle of the acquired results.
            integration_kernel: An optional Pulse object that specifies the kernel for integration. In case of multistate discrimination, a list of kernels.
            integration_kernel_parameters: An optional dictionary that contains pulse parameters for the integration kernel. In case of multistate discrimination, a list of kernels.
            integration_length: An optional float that specifies the integration length.
            measure_signal: An optional string that specifies the signal to measure.
            measure_pulse: An optional Pulse object that specifies the readout pulse for measurement.

                If this parameter is not supplied, no pulse will be played back for the measurement,
                which enables CW spectroscopy on SHFQA instruments.

            measure_pulse_length: An optional float that specifies the length of the measurement pulse.
            measure_pulse_parameters: An optional dictionary that contains parameters for the measurement pulse.
            measure_pulse_amplitude: An optional float that specifies the amplitude of the measurement pulse.
            acquire_delay: An optional float that specifies the delay between the acquisition and the measurement.
            reset_delay: An optional float that specifies the delay after the acquisition to allow for state relaxation or signal processing.
        """
        if not (isinstance(acquire_signal, str)):
            raise TypeError("`acquire_signal` must be a string.")

        if measure_signal is None:
            self.acquire(
                signal=acquire_signal,
                handle=handle,
                length=integration_length,
            )

        elif isinstance(measure_signal, str):
            self.play(
                signal=measure_signal,
                pulse=measure_pulse,
                amplitude=measure_pulse_amplitude,
                length=measure_pulse_length,
                pulse_parameters=measure_pulse_parameters,
            )

            if acquire_delay is not None:
                self.delay(
                    signal=acquire_signal,
                    time=acquire_delay,
                )

            self.acquire(
                signal=acquire_signal,
                handle=handle,
                kernel=integration_kernel,
                length=integration_length,
                pulse_parameters=integration_kernel_parameters,
            )

        if reset_delay is not None:
            self.delay(
                signal=acquire_signal,
                time=reset_delay,
            )

    def delay(
        self,
        signal: str,
        time: float | Parameter,
        precompensation_clear: bool | None = None,
    ):
        """Adds a delay on the signal with a specified time.

        Arguments:
            signal: Unique identifier of the signal where the delay should be applied.
            time: Duration of the delay.
            precompensation_clear: Clear the precompensation filter during the delay.
        """
        self.add(
            Delay(signal=signal, time=time, precompensation_clear=precompensation_clear)
        )

    def call(self, func_name, **kwargs):
        """Function call.

        Arguments:
            func_name (str | Callable): Function that should be called.
            kwargs (dict): Arguments of the function call.
        """
        self.add(Call(func_name=func_name, **kwargs))


@classformatter
@attrs.define
class AcquireLoopNt(Section):
    """Near time acquire loop.

    !!! version-changed "Deprecated in 2.14"
        Use `.sweep` outside of an `acquire_loop_rt` instead.
        For example:

        ``` py
        param = SweepParameter(values=[1, 2, 3])
        with exp.sweep(param):  # <-- outer near-time sweep
            with exp.acquire_loop_rt(count=2):  # <-- inner real-time sweep
                ...
        ```

    Attributes:
        averaging_mode (AveragingMode):
            Averaging method. One of sequential, cyclic or single shot.
            Default: [AveragingMode.CYCLIC][laboneq.core.types.enums.averaging_mode.AveragingMode].
        count (int):
            Number of loops to perform.

    [AcquireLoopNt][laboneq.dsl.experiment.section.AcquireLoopNt] inherits
    all the attributes of
    [Section][laboneq.dsl.experiment.section.Section].

    The execution type of [AcquireLoopNt][laboneq.dsl.experiment.section.AcquireLoopNt]
    sections is always
    [ExecutionType.NEAR_TIME][laboneq.core.types.enums.execution_type.ExecutionType]
    and should not be altered.
    """

    # Averaging method. One of sequential, cyclic and single_shot.
    averaging_mode: AveragingMode = attrs.field(default=AveragingMode.CYCLIC)
    # Number of loops.
    count: int | None = attrs.field(default=None)
    execution_type: ExecutionType = attrs.field(default=ExecutionType.NEAR_TIME)

    def __attrs_post_init__(self):
        warnings.warn(
            "AcquireLoopNt and acquire_loop_nt are deprecated and may be"
            " removed in a future version of LabOne Q. Use a sweep outside"
            " of the acquire_loop_rt instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__attrs_post_init__()


@classformatter
@attrs.define
class AcquireLoopRt(Section):
    """Real time acquire loop.

    Attributes:
        acquisition_type (AcquisitionType):
            Type of the acquisition. One of integration, spectroscopy,
            discrimination and RAW.
            Default: [AcquisitionType.INTEGRATION][laboneq.core.types.enums.AcquisitionType.INTEGRATION].
        averaging_mode (AveragingMode):
            Averaging method. One of sequential, cyclic or single shot.
            Default: [AveragingMode.CYCLIC][laboneq.core.types.enums.averaging_mode.AveragingMode].
        count (int):
            Number of loops to perform.
        repetition_mode (RepetitionMode):
            Repetition method. One of fastest, constant and auto.
            Default: [RepetitionMode.FASTEST][laboneq.core.types.enums.repetition_mode.RepetitionMode]
        repetition_time (float | None):
            The repetition time, when `repetition_mode` is
            [RepetitionMode.CONSTANT][laboneq.core.types.enums.repetition_mode.RepetitionMode].
        reset_oscillator_phase (bool):
            When true, reset all oscillators at the start of every step.
            Default: `False`.

    [AcquireLoopRt][laboneq.dsl.experiment.section.AcquireLoopRt] inherits
    all the attributes of
    [Section][laboneq.dsl.experiment.section.Section].

    The execution type of [AcquireLoopRt][laboneq.dsl.experiment.section.AcquireLoopNt]
    sections is always
    [ExecutionType.REAL_TIME][laboneq.core.types.enums.execution_type.ExecutionType]
    and should not be altered.
    """

    # Type of the acquisition. One of integration trigger, spectroscopy, discrimination, demodulation and RAW. The default acquisition type is INTEGRATION.
    acquisition_type: AcquisitionType = attrs.field(default=AcquisitionType.INTEGRATION)
    # Averaging method. One of sequential, cyclic and single_shot.
    averaging_mode: AveragingMode = attrs.field(default=AveragingMode.CYCLIC)
    # Number of loops.
    count: int | None = attrs.field(default=None)
    execution_type: ExecutionType = attrs.field(default=ExecutionType.REAL_TIME)
    # Repetition method. One of fastest, constant and auto.
    repetition_mode: RepetitionMode = attrs.field(default=RepetitionMode.FASTEST)
    # The repetition time, when `repetition_mode` is
    # [RepetitionMode.CONSTANT][laboneq.core.types.enums.repetition_mode.RepetitionMode.CONSTANT].
    repetition_time: float | None = attrs.field(default=None)
    # When True, reset all oscillators at the start of every step.
    reset_oscillator_phase: bool = attrs.field(default=False)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.repetition_mode == RepetitionMode.CONSTANT:
            if self.repetition_time is None:
                raise LabOneQException(
                    f"AcquireLoopRt with uid {self.uid} has RepetitionMode.CONSTANT but repetition_time is not set"
                )


@classformatter
@attrs.define
class Sweep(Section):
    """Sweep loops.

    Sweeps are used to sample through a range of parameter values.

    Attributes:
        parameters (list[Parameter] | Parameter):
            Parameters that should be swept.
            Default: `[]`.
        reset_oscillator_phase (bool):
            When True, reset all oscillators at the start of every step.
            Default: `False`.
        chunk_count (int):
            Split the sweep into N chunks.
            Default: `1`.

    [Sweep][laboneq.dsl.experiment.section.Sweep] inherits
    all the attributes of
    [Section][laboneq.dsl.experiment.section.Section].

    !!! version-changed "Changed in 2.24.0"
        `parameters` now accepts a single `Parameter` in addition to accepting a list.

    """

    # Parameters that should be swept.
    parameters: list[Parameter] = attrs.field(factory=list)
    # When True, reset all oscillators at the start of every step.
    reset_oscillator_phase: bool = attrs.field(default=False)
    # When non-zero, split the sweep into N chunks.
    chunk_count: int = attrs.field(default=1)

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.parameters is None:
            self.parameters = []
        else:
            self.parameters = (
                [self.parameters]
                if isinstance(self.parameters, Parameter)
                else list(self.parameters)
            )


@validating_allowed_values(
    {
        "alignment": [SectionAlignment.LEFT],
        "execution_type": [ExecutionType.REAL_TIME],
    }
)
@classformatter
@attrs.define
class Match(Section):
    """Execute one of the child branches depending on condition.

    Attributes:
        handle (str | None):
            Handle from which to obtain results.
            See [Section.measure][laboneq.dsl.experiment.section.Section.measure]
            and [Section.acquire][laboneq.dsl.experiment.section.Section.acquire]
            for where handles are specified.
        user_register (int | None):
            User register on which to match.
        prng_sample (PRNGSample | None):
            PRNG sample to match.
        sweep_parameter (SweepParameter | None):
            Sweep parameter to match.
        local (bool):
            Whether to fetch the codeword via the PQSC (`False`),
            SHFQC-internal bus (`True`) or automatic (`None`).
            Default: `None`.

    [Match][laboneq.dsl.experiment.section.Match] inherits
    all the attributes of
    [Section][laboneq.dsl.experiment.section.Section].

    Only subsections of type [Case][laboneq.dsl.experiment.section.Case]
    may be added to a [Match][laboneq.dsl.experiment.section.Match]
    section.
    """

    # Handle from which to obtain results
    handle: str | None = None

    # User register on which to match
    user_register: int | None = None

    # PRNG sample
    prng_sample: PRNGSample | None = None

    # Sweep parameter
    sweep_parameter: Parameter | None = None

    # Whether to fetch the codeword via the PQSC (False), SHFQC-internal bus (True) or automatic (None)
    local: bool | None = None

    def add(self, case: Case):
        """Add a branch to which to switch.

        Arguments:
            case: Branch that is added.
        """
        if not isinstance(case, Case):
            raise LabOneQException(
                f"Trying to add section to section {self.uid} which is not of type 'Case'."
            )
        if any(c.state == case.state for c in self.sections):
            raise LabOneQException(
                f"A branch which matches {case.state} already exists."
            )
        super().add(case)


@validating_allowed_values(
    {
        "alignment": [SectionAlignment.LEFT],
        "execution_type": [ExecutionType.REAL_TIME],
    }
)
@classformatter
@attrs.define
class Case(Section):
    """Branch in a match section.

    Attributes:
        state (int):
            Which state value this case is for.
            Default: `0`.

    [Case][laboneq.dsl.experiment.section.Case] inherits
    all the attributes of
    [Section][laboneq.dsl.experiment.section.Section].

    A [Case][laboneq.dsl.experiment.section.Case]
    may only be added to a [Match][laboneq.dsl.experiment.section.Match]
    section and not to any other kind of section.

    Unless matching a sweep parameter, a [Case][laboneq.dsl.experiment.section.Case]
    may only contain `PlayPulse` and `Delay` operations and not other kinds of
    operations or sections.
    """

    state: int = 0

    def add(self, obj: Operation | Section):
        """Add a child to the Case section."""
        if isinstance(obj, Case):
            raise LabOneQException("Case blocks can only be added to match blocks.")
        super().add(obj)

    @classmethod
    def from_section(cls, section: Section, state: int) -> Case:
        """Convert a section to a case section.

        Arguments:
            section:
                The section to convert.
            state:
                The state the generated case is for.

        Returns:
            case:
                A case section for the specified state with the same
                contents as the original section.

        !!! note
            This method may only be used with sections that are
            of the base [Section][laboneq.dsl.experiment.section.Section].
            type. Sub-classes may not be used.
        """
        return cls(**section.__dict__, state=state)  # type: ignore


@validating_allowed_values(
    {
        "alignment": [SectionAlignment.LEFT],
        "execution_type": [ExecutionType.REAL_TIME],
    }
)
@classformatter
@attrs.define
class PRNGSetup(Section):
    """Setup and seed the pseudo random number generator."""

    prng: PRNG | None = None

    def __iter__(self):
        return iter(self.prng)


@validating_allowed_values(
    {
        "alignment": [SectionAlignment.LEFT],
        "execution_type": [ExecutionType.REAL_TIME],
    }
)
@classformatter
@attrs.define
class PRNGLoop(Section):
    prng_sample: PRNGSample | None = None
