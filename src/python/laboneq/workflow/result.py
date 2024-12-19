# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow result objects."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from laboneq.core.utilities.highlight import pygmentize
from laboneq.workflow.taskview import TaskView

if TYPE_CHECKING:
    from datetime import datetime

    from laboneq.workflow.task_wrapper import task_


class TaskResult:
    """Task execution result.

    The instance holds execution information of an task.

    Arguments:
        task: Task producing the results.
        output: Output of the task.
        input: Input parameters of the task.
        start_time: Start time of the execution.
        end_time: End time of the execution.
        index: Index of the task.
    """

    def __init__(
        self,
        task: task_,
        output: object,
        input: dict | None = None,  # noqa: A002
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        index: tuple[object] | None = None,
    ) -> None:
        self._task = task
        self._output = output
        self._input = input or {}
        self._start_time = start_time
        self._end_time = end_time
        self._index = index

    @property
    def task(self) -> task_:
        """Task producing the result."""
        return self._task

    @property
    def name(self) -> str:
        """Task name."""
        return self._task.name

    @property
    def func(self) -> Callable:
        """Underlying function."""
        return self._task.func

    @property
    @pygmentize
    def src(self) -> str:
        """Source code of the task."""
        return self._task.src

    @property
    def index(self) -> tuple[object] | None:
        """Index of the task."""
        return self._index

    @property
    def output(self) -> object:
        """Output of the task."""
        return self._output

    @property
    def input(self) -> dict:
        """Input parameters of the task."""
        return self._input

    @property
    def start_time(self) -> datetime | None:
        """Time when the task has started."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """Time when the task has ended regularly or failed."""
        return self._end_time

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskResult):
            return NotImplemented
        return (
            self._task == other._task
            and self.output == other.output
            and self.input == other.input
            and self.index == other.index
            and self.start_time == other.start_time
            and self.end_time == other.end_time
        )

    def __repr__(self) -> str:
        attrs = ", ".join(
            [
                f"name={self.name}",
                f"output={self.output}",
                f"input={self.input}",
                f"func={self.func}",
                f"index={self.index}",
            ],
        )
        return f"TaskResult({attrs})"

    def __str__(self) -> str:
        return f"TaskResult(name={self.name}, index={self.index})"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))


class WorkflowResult:
    """Workflow execution result.

    The instance holds execution information of an workflow.

    Arguments:
        name: Name of the workflow.
        output: Output of the workflow.
        input: Input parameters of the workflow.
        start_time: Start time of the execution.
        end_time: End time of the execution.
        index: Index of the workflow.
    """

    def __init__(
        self,
        name: str,
        output: Any = None,  # noqa: ANN401
        input: dict | None = None,  # noqa: A002
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        index: tuple[object] | None = None,
    ):
        self._name = name
        self._output = output
        self._input = input or {}
        self._start_time = start_time
        self._end_time = end_time
        self._index = index
        self._tasks: list[TaskResult | WorkflowResult] = []

    @property
    def name(self) -> str:
        """Name of the workflow producing the results."""
        return self._name

    @property
    def output(self) -> Any:  # # noqa: ANN401
        """Output of the workflow."""
        return self._output

    @property
    def input(self) -> dict:
        """Input of the workflow."""
        return self._input

    @property
    def index(self) -> tuple[object] | None:
        """Index of the workflow."""
        return self._index

    @property
    def tasks(self) -> TaskView:
        """Task entries of the workflow.

        The ordering of the tasks is the order of the execution.

        Tasks is a `Sequence` of tasks, however item lookup
        is modified to support the following cases:

        Example:
            ```python
            wf = my_workflow.run()
            wf.tasks["run_experiment"]  # First task of name 'run_experiment'
            wf.tasks["run_experiment", :]  # All tasks named 'run_experiment'
            wf.tasks["run_experiment", 1:5]  # Slice tasks named 'run_experiment'
            wf.tasks[0]  # First executed task
            wf.tasks[0:5]  # Slicing

            wf.tasks.unique()  # Unique task names
            ```
        """
        return TaskView(self._tasks)

    @property
    def start_time(self) -> datetime | None:
        """The time when the workflow execution has started."""
        return self._start_time

    @property
    def end_time(self) -> datetime | None:
        """The time when the workflow execution has ended regularly or failed."""
        return self._end_time

    def __str__(self) -> str:
        return f"WorkflowResult(name={self.name}, index={self.index})"

    def _repr_pretty_(self, p, cycle):  # noqa: ANN001, ANN202, ARG002
        # For Notebooks
        p.text(str(self))

    def __eq__(self, other: object):
        if not isinstance(other, WorkflowResult):
            return NotImplemented
        return (
            self.name == other.name
            and self.input == other.input
            and self.output == other.output
            and self.tasks == other.tasks
            and self.end_time == other.end_time
            and self.start_time == other.start_time
            and self.index == other.index
        )
