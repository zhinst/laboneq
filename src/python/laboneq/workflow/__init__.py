# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A package for creating workflows.

The package provides tools and building blocks to define workflows.
"""

from laboneq.workflow.blocks import (
    break_,
    elif_,
    else_,
    for_,
    if_,
    return_,
)
from laboneq.workflow.core import (
    Workflow,
    workflow,
)
from laboneq.workflow.exceptions import WorkflowError
from laboneq.workflow.executor import execution_info
from laboneq.workflow.opts import (
    TaskOptions,
    WorkflowOptions,
    option_field,
    options,
    show_fields,
    workflow_options,
    task_options,
)
from laboneq.workflow.recorder import (
    comment,
    log,
    save_artifact,
)
from laboneq.workflow.result import TaskResult, WorkflowResult
from laboneq.workflow.task_wrapper import task
from laboneq.workflow import logbook

__all__ = [
    # Decorators
    "task",
    "workflow",
    # Core
    "Workflow",
    "WorkflowResult",
    "TaskResult",
    # Options
    "options",
    "option_field",
    "WorkflowOptions",
    "workflow_options",
    "TaskOptions",
    "task_options",
    "show_fields",
    # Workflow operations
    "return_",
    "if_",
    "elif_",
    "else_",
    "for_",
    "break_",
    # Task operations
    "comment",
    "log",
    "save_artifact",
    "execution_info",
    # Errors
    "WorkflowError",
    # Sub-packages
    "logbook",
]
