# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import copy
from functools import wraps

from laboneq.dsl.experiment.context import (
    Context,
    pop_context,
    push_context,
    reversed_iter_contexts,
)
from laboneq.dsl.experiment.experiment import Experiment
from laboneq.dsl.experiment.uid_generator import UidGenerator


class ExperimentContext(Context):
    def __init__(self, experiment, calibration):
        self.experiment = experiment
        self.calibration = calibration
        self.uid_generator = UidGenerator()

    def add(self, section):
        self.experiment.sections.append(section)

    def uid(self, prefix: str) -> str:
        """Generate a unique identifier.

        Generates an identifier that is unique to this experiment
        context by appending a count to a given prefix.

        Arguments:
            prefix:
                The prefix for the unique identifier.

        Returns:
            A unique identifier.
        """
        return self.uid_generator.uid(prefix)


class ExperimentContextManager:
    def __init__(self, *, uid=None, name=None, signals=None):
        self.kwargs = {}
        if uid is not None:
            self.kwargs["uid"] = uid
        if name is not None:
            self.kwargs["name"] = name
        if signals is not None:
            self.kwargs["signals"] = signals

    def replace(self, update_kwargs=None):
        ctx_manager = copy(self)
        if update_kwargs is not None:
            ctx_manager.kwargs |= update_kwargs
        return ctx_manager

    def __enter__(self):
        """Use as a context manager to define experiment context"""
        extra_kwargs = {}
        if "name" not in self.kwargs:
            extra_kwargs["name"] = "unnamed"
        experiment = Experiment(**self.kwargs, **extra_kwargs)
        experiment_ctx = ExperimentContext(experiment, calibration=None)
        push_context(experiment_ctx)
        return experiment

    def __exit__(self, exc_type, exc_val, exc_tb):
        experiment_ctx = pop_context()
        assert isinstance(experiment_ctx, ExperimentContext)
        if experiment_ctx.calibration is not None and exc_val is None:
            experiment_ctx.experiment.set_calibration(experiment_ctx.calibration)

    def __call__(self, f):
        """Use as a decorator for a function defining the context"""

        update_kwargs = {}
        if "uid" not in self.kwargs:
            update_kwargs["uid"] = f.__name__
        if "name" not in self.kwargs:
            update_kwargs["name"] = f.__name__

        @wraps(f)
        def wrapper(*inner_args, **inner_kwargs):
            ctx_manager = self.replace(update_kwargs=update_kwargs)
            with ctx_manager as experiment:
                f(*inner_args, **inner_kwargs)
            return experiment

        return wrapper


def current_experiment_context() -> ExperimentContext | None:
    for c in reversed_iter_contexts():
        if isinstance(c, ExperimentContext):
            return c
