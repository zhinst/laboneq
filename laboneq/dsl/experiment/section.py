# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from laboneq import dsl
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import SectionAlignment
from laboneq.core.validators import validating_allowed_values
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    ExecutionType,
    RepetitionMode,
)
from laboneq.dsl.experiment.pulse import Pulse

from .acquire import Acquire
from .call import Call
from .delay import Delay
from .operation import Operation
from .play_pulse import PlayPulse
from .reserve import Reserve
from .set import Set

if TYPE_CHECKING:
    from .. import Parameter

section_id = 0


def section_id_generator():
    global section_id
    retval = f"s_{section_id}"
    section_id += 1
    return retval


@dataclass(init=True, repr=True, order=True)
class Section:
    """Representation of a section. A section is a logical concept that groups multiple operations into a single entity
    that can be though of a container. A section can either contain other sections or a list of operations (but not both
    at the same time). Operations within a section can be aligned in various ways (left, right). Sections can have a offset
    and/or a predefined length, and they can be specified to play after another section.

    .. versionchanged:: 2.0.0
        Removed `offset` member variable.
    """

    #: Unique identifier of the section.
    uid: str = field(default_factory=section_id_generator)

    #: Alignment of operations and subsections within this section.
    alignment: SectionAlignment = field(default=SectionAlignment.LEFT)

    execution_type: Optional[ExecutionType] = field(default=None)

    #: Minimal length of the section in seconds. The scheduled section might be slightly longer, as its length is rounded to the next multiple of the section timing grid.
    length: Optional[float] = field(default=None)

    #: Play after the section with the given ID.
    play_after: Optional[Union[str, List[str]]] = field(default=None)

    #: List of children. Each child may be another section or an operation.
    children: List[Union[Section, dsl.experiment.operation.Operation]] = field(
        default_factory=list, compare=False
    )

    #: Optional trigger pulses to play during this section. See :meth:`~.Experiment.section`.
    trigger: Dict[str, Dict] = field(default_factory=dict)

    #: Whether to escalate to the system grid even if tighter alignment is possible.
    #: See :meth:`~.Experiment.section`.
    on_system_grid: Optional[bool] = field(default=False)

    def __post_init__(self):
        if self.uid is None:
            self.uid = section_id_generator()

    def add(self, section: Section):
        """Add a subsection, a sweep or a loop to the section.

        Args:
            section: Section that is added.
        """
        # if any(filter(lambda s: s.uid == section.uid, self._sections)):
        #     raise LabOneQException(
        #         f"Trying to add section with {section.uid} to section {self.uid}, but this uid is already used by a direct child section"
        #     )
        self.children.append(section)

    def _add_operation(self, operation):
        self.children.append(operation)

    @property
    def sections(self) -> Tuple[Section]:
        """A list of subsections of this section"""
        return tuple([s for s in self.children if isinstance(s, Section)])

    @property
    def operations(self) -> Tuple[Operation]:
        """A list of operations in the section.

        Note that there may be other children of a section which are not operations but subsections."""
        return tuple([s for s in self.children if isinstance(s, Operation)])

    def set(self, path: str, value: Any):
        """Set the value of an instrument node.

        Args:
            path: Path to the node whose value should be set.
            value: Value that should be set.
        """
        self._add_operation(Set(path=path, value=value))

    def play(
        self,
        signal,
        pulse,
        amplitude=None,
        phase=None,
        increment_oscillator_phase=None,
        set_oscillator_phase=None,
        length=None,
        pulse_parameters: Optional[Dict[str, Any]] = None,
        precompensation_clear: Optional[bool] = None,
        marker: Optional[Dict[str, Any]] = None,
    ):
        """Play a pulse on a signal.

        Args:
            signal: Signal the pulse should be played on.
            pulse: Pulse that should be played on the signal.
            amplitude: Amplitude of the pulse that should be played.
            phase: Phase of the pulse that should be played.
            pulse_parameters: Dictionary with user pulse function parameters (re)binding.
            precompensation_clear: Clear the precompensation filter during the pulse.
            marker: Instruction for playing marker signals along with the pulse
        """
        self._add_operation(
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

    def reserve(self, signal):
        """Operation to reserve a signal for the active section.
        Reserving an experiment signal in a section means that if there is no
        operation defined on that signal, it is not available for other sections
        as long as the active section is scoped.

        Args:
            signal: Signal that should be reserved.
        """
        self._add_operation(Reserve(signal))

    def acquire(
        self,
        signal: str,
        handle: str,
        kernel: Pulse = None,
        length: float = None,
        pulse_parameters: Optional[Dict[str, Any]] = None,
    ):
        """Acquisition of results of a signal.

        Args:
            signal: Unique identifier of the signal where the result should be acquired.
            handle: Unique identifier of the handle that will be used to access the acquired result.
            kernel: Pulse base used for the acquisition.
            length: Integration length (only valid in spectroscopy mode).
            pulse_parameters: Dictionary with user pulse function parameters (re)binding.
        """
        self._add_operation(
            Acquire(
                signal=signal,
                handle=handle,
                kernel=kernel,
                length=length,
                pulse_parameters=pulse_parameters,
            )
        )

    def delay(
        self,
        signal: str,
        time: Union[float, Parameter],
        precompensation_clear: Optional[bool] = None,
    ):
        """Adds a delay on the signal with a specified time.

        Args:
            signal: Unique identifier of the signal where the delay should be applied.
            time: Duration of the delay.
            precompensation_clear: Clear the precompensation filter during the delay.
        """
        self._add_operation(
            Delay(signal=signal, time=time, precompensation_clear=precompensation_clear)
        )

    def call(self, func_name, **kwargs):
        """Function call.

        Args:
            func_name (Union[str, Callable]): Function that should be called.
            kwargs: Arguments of the function call.
        """
        self._add_operation(Call(func_name=func_name, **kwargs))


@dataclass(init=True, repr=True, order=True)
class AcquireLoopNt(Section):
    """Near time acquire loop."""

    #: Averaging method. One of sequential, cyclic and single_shot.
    averaging_mode: AveragingMode = field(default=AveragingMode.CYCLIC)
    #: Number of loops.
    count: int = field(default=None)
    execution_type: ExecutionType = field(default=ExecutionType.NEAR_TIME)


@dataclass(init=True, repr=True, order=True)
class AcquireLoopRt(Section):
    """Real time acquire loop."""

    #: Type of the acquisition. One of integration trigger, spectroscopy, discrimination, demodulation and RAW. The default acquisition type is INTEGRATION.
    acquisition_type: AcquisitionType = field(default=AcquisitionType.INTEGRATION)
    #: Averaging method. One of sequential, cyclic and single_shot.
    averaging_mode: AveragingMode = field(default=AveragingMode.CYCLIC)
    #: Number of loops.
    count: int = field(default=None)
    execution_type: ExecutionType = field(default=ExecutionType.REAL_TIME)
    #: Repetition method. One of fastest, constant and auto.
    repetition_mode: RepetitionMode = field(default=RepetitionMode.FASTEST)
    #: The repetition time, when :py:attr:`repetition_mode` is :py:attr:`~.RepetitionMode.CONSTANT`
    repetition_time: float = field(default=None)
    #: When True, reset all oscillators at the start of every step.
    reset_oscillator_phase: bool = field(default=False)

    def __post_init__(self):
        if self.repetition_mode == RepetitionMode.CONSTANT:
            if self.repetition_time is None:
                raise LabOneQException(
                    f"AcquireLoopRt with uid {self.uid} has RepetitionMode.CONSTANT but repetition_time is not set"
                )


@dataclass(init=True, repr=True, order=True)
class Sweep(Section):
    """Sweep loops. Sweeps are used to sample through a range of parameter values."""

    #: Parameters that should be swept.
    parameters: List[Parameter] = field(default_factory=list)
    #: When True, reset all oscillators at the start of every step.
    reset_oscillator_phase: bool = field(default=False)


@validating_allowed_values(
    {
        "alignment": [SectionAlignment.LEFT],
        "execution_type": [ExecutionType.REAL_TIME],
    }
)
@dataclass(init=True, repr=True, order=True)
class Match(Section):
    """Execute one of the child branches depending on feedback result."""

    #: Handle from which to obtain results
    handle: str = ""

    #: Whether to go via the PQSC (False) or SHFQC (True)
    local: bool = False

    def add(self, case: Case):
        """Add a branch to which to switch.

        Args:
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
@dataclass(init=True, repr=True, order=True)
class Case(Section):
    """Branch in a match/case statement"""

    state: int = 0

    def add(self, obj):
        raise LabOneQException(
            f"Trying to add object to section {self.uid}. Only ``play`` and ``delay`` are allowed."
        )
