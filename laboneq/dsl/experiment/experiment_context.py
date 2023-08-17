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


class ExperimentContext(Context):
    def __init__(self, experiment, calibration):
        self.experiment = experiment
        self.calibration = calibration

    def add(self, section):
        self.experiment.sections.append(section)


class ExperimentContextManager:
    def __init__(self, *, uid=None, signals=None):
        self.kwargs = {}
        if uid is not None:
            self.kwargs["uid"] = uid
        if signals is not None:
            self.kwargs["signals"] = signals

    def replace(self, update_kwargs=None):
        ctx_manager = copy(self)
        if update_kwargs is not None:
            ctx_manager.kwargs |= update_kwargs
        return ctx_manager

    def __enter__(self):
        """Use as a context manager to define experiment context"""
        experiment = Experiment(**self.kwargs)
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
