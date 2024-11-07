# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A module that defines a view object for workflow tasks."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from laboneq.workflow import WorkflowResult
    from laboneq.workflow.result import TaskResult


class TaskView(Sequence):
    """A view of tasks.

    This class provides a view into tasks.

    Arguments:
        tasks: List of tasks.

    The class is a `Sequence` of tasks, however item lookup
    is modified to support the following cases:

        - Lookup by index and slicing
        - Lookup by name (string)
        - Lookup by name and slicing
    """

    def __init__(self, tasks: list[TaskResult | WorkflowResult] | None = None) -> None:
        self._tasks = tasks or []

    def unique(self) -> list[str]:
        """Return unique names of the tasks."""
        return list(dict.fromkeys(t.name for t in self._tasks))

    def __repr__(self) -> str:
        return repr(self._tasks)

    def __str__(self) -> str:
        return ", ".join([str(t) for t in self._tasks])

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))

    @overload
    def __getitem__(self, item: tuple[str, int]) -> TaskResult | WorkflowResult: ...

    @overload
    def __getitem__(
        self, item: tuple[str, slice] | slice
    ) -> list[TaskResult | WorkflowResult]: ...

    @overload
    def __getitem__(self, item: str | int) -> TaskResult | WorkflowResult: ...

    def __getitem__(
        self,
        item: str | int | tuple[str, int | slice] | slice,
    ) -> TaskResult | WorkflowResult | list[TaskResult | WorkflowResult]:
        """Get a single or multiple tasks.

        Arguments:
            item: Index, name of the task, slice or a tuple.

                If index or name is given, the return value will be a single object,
                the first one found in the sequence.

                tuple: A tuple of format (<name>, <slice>) will return
                    list of tasks.

                tuple: A tuple of format (<name>, <index>) will return
                    a single task in the given index.

        Returns:
            Task or list of tasks, depending on the input filter.

        Raises:
            KeyError: Task by name was not found.
            IndexError: Task by index was not found.
        """
        if isinstance(item, str):
            try:
                return next(t for t in self._tasks if t.name == item)
            except StopIteration:
                raise KeyError(item) from None
        if isinstance(item, tuple):
            items = [t for t in self._tasks if t.name == item[0]]
            if not items:
                raise KeyError(item[0])
            return items[item[1]]
        return self._tasks[item]

    def __len__(self) -> int:
        return len(self._tasks)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, type(self)):
            return self._tasks == other._tasks
        return NotImplemented
