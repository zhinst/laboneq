# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import copy
from functools import wraps

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import (
    AcquisitionType,
    AveragingMode,
    RepetitionMode,
)
from laboneq.dsl import Parameter
from laboneq.dsl.experiment.context import (
    Context,
    peek_context,
    pop_context,
    push_context,
)
from laboneq.dsl.experiment.experiment_context import (
    current_experiment_context,
)
from laboneq.dsl.experiment.section import (
    AcquireLoopNt,
    AcquireLoopRt,
    Case,
    Match,
    Section,
    Sweep,
    PRNGSetup,
    PRNGLoop,
)
from laboneq.dsl.experiment.uid_generator import GLOBAL_UID_GENERATOR
from laboneq.dsl.prng import PRNGSample


class SectionContext(Context):
    def __init__(self, section, auto_add):
        self.section = section
        self.auto_add = auto_add

    def add(self, section):
        self.section.add(section)


class SectionContextManagerBase:
    section_class = ...

    def __init__(self, kwargs, auto_add=True):
        self.kwargs = kwargs
        self.auto_add = auto_add

    def replace(self, update_kwargs=None, auto_add=None):
        ctx_manager = copy(self)
        if update_kwargs is not None:
            ctx_manager.kwargs |= update_kwargs
        if auto_add is not None:
            ctx_manager.auto_add = auto_add
        return ctx_manager

    def _uid(self, prefix):
        context = current_experiment_context()
        if context is not None:
            return context.uid(prefix)
        return GLOBAL_UID_GENERATOR.uid(prefix)

    def _uid_name_kwargs(self):
        kwargs = self.kwargs.copy()
        name = kwargs.pop("name", None)
        if name is None:
            name = "unnamed"
        uid = kwargs.pop("uid", None)
        if uid is None:
            uid = self._uid(name)
        return uid, name, kwargs

    def _section_create(self):
        uid, name, kwargs = self._uid_name_kwargs()
        return self.section_class(uid=uid, name=name, **kwargs)

    def _section_post_create(self, section, parent):
        pass

    def _peek_section_parent(self):
        parent = peek_context()
        if isinstance(parent, SectionContext):
            return parent.section
        return None

    def __enter__(self):
        section = self._section_create()
        parent = self._peek_section_parent()
        self._section_post_create(section, parent)
        section_ctx = SectionContext(section, auto_add=self.auto_add)
        push_context(section_ctx)
        return section

    def __exit__(self, exc_type, exc_val, exc_tb):
        section_ctx = pop_context()
        assert isinstance(section_ctx, SectionContext)
        if exc_val is None and section_ctx.auto_add:
            # auto-add section to parent
            parent = peek_context()
            if parent is not None:
                parent.add(section_ctx.section)


class SectionContextManager(SectionContextManagerBase):
    section_class = Section

    def __init__(
        self,
        length=None,
        alignment=None,
        uid=None,
        name=None,
        on_system_grid=None,
        play_after: str | list[str] | None = None,
        trigger: dict[str, dict[str, int]] | None = None,
        execution_type=None,
    ):
        kwargs = {}
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        if length is not None:
            kwargs["length"] = length
        if alignment is not None:
            kwargs["alignment"] = alignment
        if play_after is not None:
            kwargs["play_after"] = play_after
        if trigger is not None:
            kwargs["trigger"] = trigger
        if on_system_grid is not None:
            kwargs["on_system_grid"] = on_system_grid
        if execution_type is not None:
            kwargs["execution_type"] = execution_type
        super().__init__(kwargs=kwargs)

    def __call__(self, f):
        """Use as a decorator for a function defining the context"""

        update_kwargs = {}
        if "uid" not in self.kwargs:
            update_kwargs["uid"] = f.__name__

        @wraps(f)
        def wrapper(*inner_args, section_auto_add=True, **inner_kwargs):
            ctx_manager = self.replace(
                update_kwargs=update_kwargs, auto_add=section_auto_add
            )
            with ctx_manager as section:
                f(*inner_args, **inner_kwargs)
            return section

        return wrapper


class SweepSectionContextManager(SectionContextManagerBase):
    section_class = Sweep

    def __init__(
        self,
        parameters,
        uid=None,
        name=None,
        alignment=None,
        reset_oscillator_phase=False,
        chunk_count=1,
    ):
        kwargs = dict(parameters=parameters, chunk_count=chunk_count)
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        if alignment is not None:
            kwargs["alignment"] = alignment
        if reset_oscillator_phase is not None:
            kwargs["reset_oscillator_phase"] = reset_oscillator_phase
        super().__init__(kwargs=kwargs)

    def __enter__(self):
        section = super().__enter__()
        if len(section.parameters) == 1:
            return section.parameters[0]
        return tuple(section.parameters)


class AcquireLoopNtSectionContextManager(SectionContextManagerBase):
    section_class = AcquireLoopNt

    def __init__(self, count, averaging_mode=AveragingMode.CYCLIC, uid=None, name=None):
        kwargs = dict(
            count=count,
            averaging_mode=averaging_mode,
        )
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        super().__init__(kwargs=kwargs)


class AcquireLoopRtSectionContextManager(SectionContextManagerBase):
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
        name=None,
    ):
        kwargs = dict(
            count=count,
            averaging_mode=averaging_mode,
            repetition_mode=repetition_mode,
            repetition_time=repetition_time,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=reset_oscillator_phase,
        )
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        super().__init__(kwargs=kwargs)


class MatchSectionContextManager(SectionContextManagerBase):
    section_class = Match

    def __init__(
        self,
        handle: str | None = None,
        user_register: int | None = None,
        prng_sample: PRNGSample | None = None,
        sweep_parameter: Parameter | None = None,
        uid=None,
        name=None,
        play_after=None,
        local=None,
    ):
        kwargs = {}
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        if play_after is not None:
            kwargs["play_after"] = play_after
        if handle is not None:
            kwargs["handle"] = handle
        if user_register is not None:
            kwargs["user_register"] = user_register
        if prng_sample is not None:
            kwargs["prng_sample"] = prng_sample
        if sweep_parameter is not None:
            kwargs["sweep_parameter"] = sweep_parameter
        if local is not None:
            kwargs["local"] = local
        super().__init__(kwargs=kwargs)


class CaseSectionContextManager(SectionContextManagerBase):
    section_class = Case

    def __init__(self, uid, name, state):
        kwargs = dict(state=state)
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        super().__init__(kwargs=kwargs)

    def _section_post_create(self, section, parent):
        super()._section_post_create(section, parent)
        if not isinstance(parent, Match):
            raise LabOneQException("Case section must be inside a Match section")


class PRNGSetupContextManager(SectionContextManagerBase):
    section_class = PRNGSetup

    def __init__(self, prng, uid, name):
        kwargs = {"prng": prng}
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        super().__init__(kwargs=kwargs)

    def __enter__(self):
        section = super().__enter__()
        return section.prng


class PRNGLoopContextManager(SectionContextManagerBase):
    section_class = PRNGLoop

    def __init__(self, prng_sample, uid, name):
        kwargs = {"prng_sample": prng_sample}
        if uid is not None:
            kwargs["uid"] = uid
        if name is not None:
            kwargs["name"] = name
        super().__init__(kwargs=kwargs)

    def __enter__(self):
        section = super().__enter__()
        return section.prng_sample


def current_section_context() -> SectionContext | None:
    context = peek_context()
    if context is None or not isinstance(context, SectionContext):
        return None
    return context
