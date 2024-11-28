# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow break loop block."""

from __future__ import annotations

from laboneq.workflow import blocks, variable_tracker
from laboneq.workflow.exceptions import WorkflowError
from laboneq.workflow.executor import ExecutionStatus, ExecutorState


class BreakLoopBlock(blocks.Block):
    """Break loop block.

    A block that breaks out of the currently running innermost workflow loop.
    """

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        executor.set_block_status(self, ExecutionStatus.FINISHED)
        executor.interrupt_loop()

    def __str__(self):
        return "break_()"


@variable_tracker.track
def break_() -> None:
    """Break statement to break out of the currently running innermost loop.

    The equivalent of Python's `break` statement.

    Raises:
        WorkflowError: Defined outside of loop scope.
    """
    ctx = blocks.BlockBuilderContext.get_active()
    if ctx:
        # Go upwards in stack so that an loop would be encountered first if one exists.
        for root in blocks.BlockBuilderContext.iter_stack(reverse=True):
            if isinstance(root, blocks.ForExpression):
                ctx.extend(BreakLoopBlock())
                return
            if isinstance(root, blocks.WorkflowBlock):
                break
        msg = "A `break_` statement may only occur within a `for_` loop"
        raise WorkflowError(msg)
