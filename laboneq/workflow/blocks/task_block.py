# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow block for tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.workflow.timestamps import utc_now
from laboneq.workflow.blocks.block import Block
from laboneq.workflow.executor import ExecutionStatus, ExecutorState
from laboneq.workflow.reference import Reference
from laboneq.workflow.result import TaskResult

if TYPE_CHECKING:
    from laboneq.workflow.opts import TaskOptions
    from laboneq.workflow.task_wrapper import task_


class TaskBlock(Block):
    """Task block.

    `TaskBlock` is an workflow executor for a task.

    Arguments:
        task: A task this block contains.
        parameters: Input parameters of the task.
    """

    def __init__(self, task: task_, parameters: dict | None = None):
        super().__init__(parameters=parameters)
        self.task = task
        self._ref = Reference(self)

    @property
    def ref(self) -> Reference:
        """Reference to the object."""
        return self._ref

    @property
    def hidden(self) -> bool:
        """Whether or not the task is a hidden task."""
        return self.task._hidden

    @property
    def options_type(self) -> type[TaskOptions] | None:
        """Type of block options."""
        return self.task._options

    @property
    def name(self) -> str:
        """Name of the task."""
        return self.task.name

    def execute(self, executor: ExecutorState) -> None:
        """Execute the task."""
        if self.hidden:
            self._exucute_hidden(executor)
        else:
            self._execute_visible(executor)

    def _exucute_hidden(self, executor: ExecutorState) -> None:
        params = {}
        if self.parameters:
            params = executor.resolve_inputs(self)
            if self.options_type and params.get("options") is None:
                params["options"] = executor.get_options(self.name)
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        try:
            executor.set_variable(self.ref, self.task.func(**params))
        finally:
            executor.set_block_status(self, ExecutionStatus.FINISHED)

    def _execute_visible(self, executor: ExecutorState) -> None:
        params = {}
        if self.parameters:
            params = executor.resolve_inputs(self)
            if self.options_type and params.get("options") is None:
                params["options"] = executor.get_options(self.name)
        task = TaskResult(
            task=self.task,
            output=None,
            input=params,
            index=executor.get_index(),
            start_time=utc_now(),
        )
        executor.recorder.on_task_start(task)
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        try:
            task._output = self.task.func(**params)
        except Exception as error:
            task._end_time = utc_now()
            executor.recorder.on_task_error(task, error)
            raise
        finally:
            task._end_time = utc_now()
            executor.set_block_status(self, ExecutionStatus.FINISHED)
            executor.add_task_result(task)
            executor.recorder.on_task_end(task)
        executor.set_variable(self.ref, task.output)
        if executor.settings.run_until == self.name:
            executor.interrupt()

    def __repr__(self):
        return f"TaskBlock(task={self.task}, parameters={self.parameters})"

    def __str__(self):
        return f"task(name={self.name})"
