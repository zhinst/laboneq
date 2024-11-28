# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Options for workflows."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, Type

from laboneq.workflow.opts.options_base import (
    BaseOptions,
    _options_decorator,
    option_field,
    options,
)

if TYPE_CHECKING:
    from laboneq.workflow.logbook import LogbookStore


@options
class TaskOptions:
    """Base class for task options."""


def _logstore_converter(x: object) -> object:
    # To avoid circular import
    from laboneq.workflow.logbook import LogbookStore

    return [x] if isinstance(x, LogbookStore) else x


@options
class WorkflowOptions:
    """Base options for a workflow.

    Attributes:
        logstore:
            The logstore to use for the particular workflow.
            If left as `None`, uses the currently activated logstores.
            The field accepts either a single logstore or a list of logstores.
            Empty list results to no logging at all.

            This field is not serialized/deserialized.
        _task_options:
            A mapping of sub-task and sub-workflow options.
            A task can have only one unique set of options per workflow.
    """

    logstore: list[LogbookStore] | LogbookStore | None = option_field(
        None,
        description="The logstores to use.",
        exclude=True,
        repr=False,
        converter=_logstore_converter,
    )
    _task_options: dict[str, BaseOptions] = option_field(
        factory=dict, description="task options", alias="_task_options"
    )


def workflow_options(
    cls: Type | None = None, *, base_class: Type[WorkflowOptions] = WorkflowOptions
) -> type:
    """Decorator to make a class a workflow options class.

    Args:
        cls:
            The class to decorate
        base_class:
            The base class for the decorated class, default is `WorkflowOptions`.
            Must be a subclass of `WorkflowOptions`.
    """

    @wraps(cls)
    def wrapper(cls):
        return options(cls=cls, base_class=base_class)

    return _options_decorator(cls, base_class, WorkflowOptions, wrapper)


def task_options(
    cls: Type | None = None, *, base_class: Type[WorkflowOptions] = TaskOptions
) -> type:
    """Decorator to make a class a task options class.

    cls:
        The class to decorate
    base_class:
        The base class for the decorated class, default is `WorkflowOptions`.
        Must be a subclass of `TaskOptions`.
    """

    @wraps(cls)
    def wrapper(cls):
        return options(cls=cls, base_class=base_class)

    return _options_decorator(cls, base_class, TaskOptions, wrapper)
