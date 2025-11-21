# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow recorder that records different events during execution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from laboneq.workflow.timestamps import utc_now

if TYPE_CHECKING:
    from laboneq.workflow.result import TaskResult, WorkflowResult
    from laboneq.workflow.typing import SimpleDict


class Artifact:
    """An artifact to record.

    An artifact consists of a Python object that a workflow wishes to
    record plus the additional information required to store and
    reference it.

    Arguments:
        name:
            A name hint for the artifact. Logbooks may use this to generate
            meaningful filenames for artifacts when they are saved to disk,
            for example.
        obj:
            The object to be recorded.
        metadata:
            Additional metadata for the artifact (optional).
        options:
            Serialization options for the artifact (optional).
    """

    def __init__(
        self,
        name: str,
        obj: object,
        metadata: SimpleDict | None = None,
        options: SimpleDict | None = None,
    ):
        self.name = name
        self.obj = obj
        self.metadata = metadata or {}
        self.options = options or {}
        self.timestamp = utc_now()


class ExecutionRecorder(Protocol):
    """A class that defines interface for an execution recorder.

    The recorder provides an interface to record specific actions
    during the execution.
    """

    def on_start(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution starts."""

    def on_end(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution ends."""

    def on_error(
        self,
        workflow_result: WorkflowResult,
        error: Exception,
    ) -> None:
        """Called when the workflow raises an exception."""

    def on_task_start(
        self,
        task: TaskResult,
    ) -> None:
        """Called when a task begins execution."""

    def on_task_end(
        self,
        task: TaskResult,
    ) -> None:
        """Called when a task ends execution."""

    def on_task_error(
        self,
        task: TaskResult,
        error: Exception,
    ) -> None:
        """Called when a task raises an exception."""

    def comment(
        self,
        message: str,
    ) -> None:
        """Called to leave a comment."""

    def log(
        self,
        level: int,
        message: str,
        *args: object,
    ) -> None:
        """Called to leave a log message."""

    def save(
        self,
        artifact: Artifact,
    ) -> None:
        """Called to record an artifact.

        Arguments:
            artifact:
                The artifact to be saved.
        """


class ExecutionRecorderManager(ExecutionRecorder):
    """A class that manages multiple execution recorders.

    When an error is recorded via specific methods, a single error is broadcasted only
    once for each recorder. This means that if an error is recorded multiple times,
    only the first time is forwarded to the recorders.
    """

    def __init__(self) -> None:
        self._recorders: list[ExecutionRecorder] = []

    def _maybe_error_recorded(self, error: Exception) -> bool:
        """Check whether the error was already recorded.

        Returns:
            True: Error was recorded
            False: Error was not previously recorded, but is now labeled as such.
        """
        if not getattr(
            error,
            "_is_recorded",
            False,
        ):
            error._is_recorded = True
            return False
        return True

    def add_recorder(self, recorder: ExecutionRecorder) -> None:
        """Add a recorder to the execution.

        Arguments:
            recorder: A recorder that records the execution information.
        """
        self._recorders.append(recorder)

    def on_start(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution starts."""
        for recorder in self._recorders:
            recorder.on_start(workflow_result)

    def on_end(self, workflow_result: WorkflowResult) -> None:
        """Called when the workflow execution ends."""
        for recorder in self._recorders:
            recorder.on_end(workflow_result)

    def on_error(
        self,
        workflow_result: WorkflowResult,
        error: Exception,
    ) -> None:
        """Called when the workflow raises an exception."""
        if not self._maybe_error_recorded(error):
            for recorder in self._recorders:
                recorder.on_error(workflow_result, error)

    def on_task_start(
        self,
        task: TaskResult,
    ) -> None:
        """Called when a task begins execution."""
        for recorder in self._recorders:
            recorder.on_task_start(task)

    def on_task_end(self, task: TaskResult) -> None:
        """Add a task result."""
        for recorder in self._recorders:
            recorder.on_task_end(task)

    def on_task_error(
        self,
        task: TaskResult,
        error: Exception,
    ) -> None:
        """Called when a task raises an exception."""
        if not self._maybe_error_recorded(error):
            for recorder in self._recorders:
                recorder.on_task_error(task, error)

    def comment(
        self,
        message: str,
    ) -> None:
        """Called to leave a comment."""
        for recorder in self._recorders:
            recorder.comment(message)

    def log(
        self,
        level: int,
        message: str,
        *args: object,
    ) -> None:
        """Called to leave a log message."""
        for recorder in self._recorders:
            recorder.log(level, message, *args)

    def save(
        self,
        artifact: Artifact,
    ) -> None:
        """Called to record an artifact.

        Arguments:
            artifact:
                The artifact to be saved.
        """
        for recorder in self._recorders:
            recorder.save(artifact)
