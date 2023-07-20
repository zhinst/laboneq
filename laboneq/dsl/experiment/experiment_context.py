# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import wraps

from laboneq.dsl.experiment.context import (
    Context,
    iter_contexts,
    pop_context,
    push_context,
)
from laboneq.dsl.experiment.experiment import Experiment


class ExperimentContext(Context):
    def __init__(self, *, uid=None, signals=None):
        self.kwargs = {}
        if uid is not None:
            self.kwargs["uid"] = uid
        if signals is not None:
            self.kwargs["signals"] = signals

        self.calibration = None
        self.experiment = None

    def __enter__(self):
        """Use as a context manager to define experiment context"""

        push_context(self)
        self.experiment = Experiment(**self.kwargs)
        return self.experiment

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert pop_context() is self
        if self.calibration is not None and exc_val is None:
            self.experiment.set_calibration(self.calibration)

    def __call__(self, f):
        """Use as a decorator for a function defining the context"""

        if "uid" not in self.kwargs:
            self.kwargs["uid"] = f.__name__

        @wraps(f)
        def wrapper(*inner_args, **inner_kwargs):
            with self:
                f(*inner_args, **inner_kwargs)
            return self.experiment

        return wrapper

    def add(self, section):
        self.experiment.sections.append(section)


def current_experiment_context() -> ExperimentContext | None:
    for c in iter_contexts():
        if isinstance(c, ExperimentContext):
            return c
