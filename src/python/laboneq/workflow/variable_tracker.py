# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A function variable tracker for variables at workflow definition time."""

from __future__ import annotations

import inspect
from collections import defaultdict
from functools import wraps
from typing import Callable, TypeVar

from laboneq.workflow import reference
from laboneq.workflow._context import LocalContext


class WorkflowFunctionVariableTracker:
    """Workflow function variable tracker."""

    def __init__(self):
        self._variables: dict[str, list] = defaultdict(list)

    def add_variable(self, name: str, value: object) -> None:
        """Register a local scope variable to the tracker.

        Arguments:
            name: Name of the variable
            value: Value of the variable
        """
        if not isinstance(value, reference.Reference):
            value = reference.Reference(None, default=value)
        self._variables[name].append(value)


class WorkflowFunctionVariableTrackerContext(
    LocalContext[WorkflowFunctionVariableTracker]
):
    """Workflow function variable tracker context."""

    _scope = "variable_tracker"

    @classmethod
    def exit(cls) -> WorkflowFunctionVariableTracker | None:
        """Exit the context and map overwritten references."""
        tracker = super().exit()
        if tracker:
            for var in tracker._variables.values():
                for ref in reversed(var[:-1]):
                    reference.add_overwrite(var[-1], ref)
            return tracker
        return None


T = TypeVar("T", bound=Callable)


def track(func: T) -> T:
    """A decorator to mark a callable to track local scope variables.

    When a callable with decorated, the tracker will look up one level
    (the wrapped callable's caller) local scope variables.

    The variables before the wrapped function within the scope are recorded.
    This means the last defined variable of any function scope wont be recorded
    if it is not used within another wrapped callable. Excluding the last, unused
    variable is fine since it has no workflow object consumes it.
    """

    @wraps(func)
    def inner(*args, **kwargs):  # noqa: ANN202
        ctx = WorkflowFunctionVariableTrackerContext.get_active()
        if ctx:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                for k, v in frame.f_back.f_locals.items():
                    ctx.add_variable(k, v)
        return func(*args, **kwargs)

    return inner  # type: ignore  # noqa: PGH003
