# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import wraps

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import (
    AcquisitionType,
    AveragingMode,
    ExecutionType,
    RepetitionMode,
)
from laboneq.dsl.experiment.context import (
    Context,
    current_context,
    peek_context,
    pop_context,
    push_context,
)
from laboneq.dsl.experiment.section import (
    AcquireLoopNt,
    AcquireLoopRt,
    Case,
    Match,
    Section,
    Sweep,
)


class SectionContextBase(Context):
    section_class = ...

    def __init__(self):
        self.section = None
        self.kwargs = {}
        self._auto_add = True

    def __enter__(self):
        self.section = self.section_class(**self.kwargs)
        parent = current_context()
        if self.section.execution_type is None:
            if parent is not None:
                if isinstance(parent, SectionContextBase):
                    self.section.execution_type = parent.section.execution_type
        elif self.section.execution_type == ExecutionType.NEAR_TIME:
            if parent is not None and isinstance(parent, SectionContextBase):
                if parent.section.execution_type == ExecutionType.REAL_TIME:
                    raise LabOneQException(
                        "Cannot nest near-time section inside real-time context"
                    )
        if self.section.execution_type is None:
            self.section.execution_type = ExecutionType.NEAR_TIME
        push_context(self)
        return self.section

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert pop_context() is self
        if exc_val is None and self._auto_add:
            # auto-add section to parent
            parent = current_context()
            if parent is not None:
                parent.add(self.section)

    def __call__(self, f):
        raise NotImplementedError

    def add(self, section):
        self.section.add(section)


class SectionSectionContext(SectionContextBase):
    section_class = Section

    def __init__(
        self,
        length=None,
        alignment=None,
        uid=None,
        on_system_grid=None,
        play_after: str | list[str] | None = None,
        trigger: dict[str, dict[str, int]] | None = None,
        execution_type=None,
    ):
        super().__init__()
        if uid is not None:
            self.kwargs["uid"] = uid
        if length is not None:
            self.kwargs["length"] = length
        if alignment is not None:
            self.kwargs["alignment"] = alignment
        if play_after is not None:
            self.kwargs["play_after"] = play_after
        if trigger is not None:
            self.kwargs["trigger"] = trigger
        if on_system_grid is not None:
            self.kwargs["on_system_grid"] = on_system_grid
        if execution_type is not None:
            self.kwargs["execution_type"] = execution_type

    def __call__(self, f):
        """Use as a decorator for a function defining the context"""

        if "uid" not in self.kwargs:
            self.kwargs["uid"] = f.__name__

        @wraps(f)
        def wrapper(*inner_args, section_auto_add=True, **inner_kwargs):
            self._auto_add = bool(section_auto_add)
            with self:
                f(*inner_args, **inner_kwargs)
            return self.section

        return wrapper

    def add(self, section):
        self.section.add(section)


class SweepSectionContext(SectionContextBase):
    section_class = Sweep

    def __init__(
        self,
        parameters,
        execution_type=None,
        uid=None,
        alignment=None,
        reset_oscillator_phase=False,
        chunk_count=1,
    ):
        super().__init__()
        self.kwargs = {"parameters": parameters}
        if uid is not None:
            self.kwargs["uid"] = uid
        if execution_type is not None:
            self.kwargs["execution_type"] = execution_type

        if alignment is not None:
            self.kwargs["alignment"] = alignment

        if reset_oscillator_phase is not None:
            self.kwargs["reset_oscillator_phase"] = reset_oscillator_phase

        self.kwargs["chunk_count"] = chunk_count

    def __enter__(self):
        super().__enter__()
        if len(self.section.parameters) == 1:
            return self.section.parameters[0]
        return tuple(self.section.parameters)


class AcquireLoopNtSectionContext(SectionContextBase):
    section_class = AcquireLoopNt

    def __init__(self, count, averaging_mode=AveragingMode.CYCLIC, uid=None):
        super().__init__()
        self.kwargs = dict(
            count=count,
            averaging_mode=averaging_mode,
        )
        if uid is not None:
            self.kwargs["uid"] = uid


class AcquireLoopRtSectionContext(SectionContextBase):
    section_class = AcquireLoopRt

    def __init__(
        self,
        count=None,
        averaging_mode=AveragingMode.CYCLIC,
        repetition_mode=RepetitionMode.FASTEST,
        repetition_time=None,
        acquisition_type=AcquisitionType.INTEGRATION,
        reset_oscillator_phase=False,
        uid=None,
    ):
        super().__init__()
        self.kwargs = dict(
            count=count,
            averaging_mode=averaging_mode,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=reset_oscillator_phase,
        )
        if uid is not None:
            self.kwargs["uid"] = uid


class MatchSectionContext(SectionContextBase):
    section_class = Match

    def __init__(
        self,
        handle: str | None = None,
        user_register: int | None = None,
        uid=None,
        play_after=None,
    ):
        super().__init__()
        if uid is not None:
            self.kwargs["uid"] = uid
        if play_after is not None:
            self.kwargs["play_after"] = play_after
        if handle is not None:
            self.kwargs["handle"] = handle
        if user_register is not None:
            self.kwargs["user_register"] = user_register


class CaseSectionContext(SectionContextBase):
    section_class = Case

    def __init__(
        self,
        uid,
        state,
    ):
        super().__init__()
        self.kwargs["state"] = state
        if uid is not None:
            self.kwargs["uid"] = uid

    def __enter__(self):
        if not isinstance(peek_context(), MatchSectionContext):
            raise LabOneQException("Case section must be inside a Match section")
        return super().__enter__()


def active_section() -> Section:
    s = peek_context()
    if s is None or not isinstance(s, SectionContextBase):
        raise LabOneQException("Must be in a section context")
    return s.section
