# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Task decorator to convert Python functions into workflow tasks."""

from __future__ import annotations

import inspect
import textwrap
from functools import partial, update_wrapper
from typing import Callable, Generic, TypeVar, cast, overload

from typing_extensions import ParamSpec

from laboneq.core.utilities.highlight import pygmentize
from laboneq.workflow import _utils, variable_tracker
from laboneq.workflow.blocks import BlockBuilderContext, TaskBlock
from laboneq.workflow import opts

T = ParamSpec("T")
B = TypeVar("B")


class task_(Generic[T, B]):  # noqa: N801
    """A task that wraps a Python function.

    Arguments:
        func: Function to be executed as a task.
        name: Optional name of the task.
            If `None`, the name of the function is used.
        save: A flag to indicate whether the task inputs and outputs should be
            saved by logbooks.
            The flag has no effect on saving done inside the task.
    """

    def __init__(
        self,
        func: Callable[T, B],
        name: str | None = None,
        *,
        save: bool = True,
        hidden: bool = False,
    ) -> None:
        self._func = func
        self._name: str = name if name is not None else func.__name__
        self._save = save
        self._hidden = hidden
        self.__doc__ = func.__doc__
        self._options = opts.get_and_validate_param_type(
            self._func, opts.TaskOptions, "options"
        )

    @property
    @pygmentize
    def src(self) -> str:
        """Source code of the task."""
        src = inspect.getsource(self._func)
        return textwrap.dedent(src)

    @property
    def func(self) -> Callable:
        """Underlying Python function."""
        return self._func

    @property
    def name(self) -> str:
        """The name of the task."""
        return self._name

    @property
    def save(self) -> bool:
        """Whether the task inputs and outputs should be saved by logbooks."""
        return self._save

    @variable_tracker.track
    def __call__(self, *args: T.args, **kwargs: T.kwargs) -> B:  # noqa: D102
        root = BlockBuilderContext.get_active()
        if root:
            block = TaskBlock(
                task=self,
                parameters=_utils.create_argument_map(self.func, *args, **kwargs),
            )
            root.extend(block)
            return cast(B, block.ref)
        return self._func(*args, **kwargs)

    def __repr__(self):
        return f"task(func={self.func}, name={self.name}, save={self.save})"

    def __str__(self):
        return f"task(name={self.name})"


@overload
def task(
    func: Callable[T, B],
    *,
    name: str | None = ...,
    save: bool = ...,
    hidden: bool = ...,
) -> task_[T, B]: ...


@overload
def task(
    func: None = ..., *, name: str | None = ..., save: bool = ..., hidden: bool = ...
) -> Callable[[Callable[T, B]], task_[T, B]]: ...


def task(
    func: Callable[T, B] | None = None,
    *,
    name: str | None = None,
    save: bool = True,
    hidden: bool = False,
) -> task_[T, B] | Callable[[Callable[T, B]], task_[T, B]]:
    """Mark a function as a workflow task.

    If the decorated function is used outside of an workflow related context, or
    within another task, the underlying behavior does not change.

    The function signature must not have positional-only arguments.

    Arguments:
        func: Function to be wrapped as a task.
        name: Name of the task.
            Defaults to function name.
        save: A flag to indicate whether the task inputs and outputs should be
            saved by logbooks.
            The flag has no effect on saving done inside the task.
        hidden: Mark the task as hidden task.
            When set, the task won't be visible to `LogBook`s and won't be
            included in the results.

    Example:
        ```python
        from laboneq.workflow import task


        @task
        def my_task(x, y):
            return x + y


        my_task(1, 1)
        ```
    """
    if func is None:
        return partial(task_, name=name, save=save, hidden=hidden)
    if isinstance(func, task_):
        return update_wrapper(
            task_(func=func.func, name=name, save=save, hidden=hidden),
            func.func,
        )
    return update_wrapper(
        task_(func=func, name=name, save=save, hidden=hidden),
        func,
    )
