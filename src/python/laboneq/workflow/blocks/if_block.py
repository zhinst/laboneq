# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow if block."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

from laboneq.workflow import blocks, variable_tracker
from laboneq.workflow.exceptions import WorkflowError
from laboneq.workflow.executor import ExecutionStatus, ExecutorState

if TYPE_CHECKING:
    from collections.abc import Generator


class ConditionalChain(blocks.Block):
    """A block that represents chained conditionals.

    Blocks within `ConditionalChain` are parallel to each other
    and only one, or none is executed.

    Blocks within `ConditionalChain` must mark themselves as skipped if condition is not
    fulfilled.
    """

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block.

        Exists after the first block that is marked as finished.
        """
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        for idx, block in enumerate(self.body):
            if executor.get_block_status(block) in (
                ExecutionStatus.FINISHED,
                ExecutionStatus.SKIPPED,
            ):
                continue
            block.execute(executor)
            if executor.get_block_status(block) == ExecutionStatus.FINISHED:
                # Conditional chain finished, mark the rest as skipped
                for remainder in self.body[idx:]:
                    executor.set_block_status(remainder, ExecutionStatus.SKIPPED)
                break
        executor.set_block_status(self, ExecutionStatus.FINISHED)

    def __str__(self):
        return "conditional"


class IFExpression(blocks.Block):
    """If expression.

    A block that is executed if a given `condition` is true.

    Arguments:
        condition: A condition that has to be true for block to be
            executed.
    """

    def __init__(self, condition: object) -> None:
        super().__init__(parameters={"condition": condition})

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        arg = executor.resolve_inputs(self)["condition"]
        if bool(arg):
            executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
            for block in self.body:
                if executor.get_block_status(block) in (
                    ExecutionStatus.FINISHED,
                    ExecutionStatus.SKIPPED,
                ):
                    continue
                block.execute(executor)
            executor.set_block_status(self, ExecutionStatus.FINISHED)
        else:
            executor.set_block_status(self, ExecutionStatus.SKIPPED)

    def __str__(self):
        return "if_()"


class ElseIfExpression(IFExpression):
    """Else if expression.

    A block that is executed if a given `condition` is true and preceding
    if expression is evaluated false.

    Arguments:
        condition: A condition that has to be true for block to be
            executed.
    """

    def __str__(self):
        return "elif_()"


class ElseExpression(blocks.Block):
    """Else expression.

    A block that is executed if preceding conditionals are evaluated false.
    """

    def __init__(self) -> None:
        super().__init__()

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        for block in self.body:
            if executor.get_block_status(block) in (
                ExecutionStatus.FINISHED,
                ExecutionStatus.SKIPPED,
            ):
                continue
            block.execute(executor)
        executor.set_block_status(self, ExecutionStatus.FINISHED)

    def __str__(self):
        return "else_()"


@variable_tracker.track
@contextmanager
def if_(condition: Any) -> Generator[None, None, None]:  # noqa: ANN401
    """Workflow if statement.

    The equivalent of Python's if-statement.

    Arguments:
        condition: A condition that has to be true for code block to be
            executed.

    Example:
        ```python
        from laboneq import workflow


        @workflow.workflow
        def a_workflow(x):
            with workflow.if_(x == 1):
                ...
        ```
    """
    expr = IFExpression(condition=condition)
    collection = ConditionalChain()
    root = blocks.BlockBuilderContext.get_active()
    if root:
        root.extend(collection)
    with expr.collect():
        yield
    collection.extend(expr)


@variable_tracker.track
@contextmanager
def elif_(condition: Any) -> Generator[None, None, None]:  # noqa: ANN401
    """Workflow else if statement.

    The equivalent of Python's elif-statement.

    Arguments:
        condition: A condition that has to be true for code block to be
            executed.

    Example:
        ```python
        from laboneq import workflow


        @workflow.workflow
        def a_workflow(x):
            with workflow.if_(x == 1):
                ...
            with workflow.elif_(x == 2):
                ...
        ```

    Raises:
        WorkflowError:
            Expression is defined without `if_()`
    """
    root = blocks.BlockBuilderContext.get_active()
    if not root:
        yield
        return
    if (
        not root.body
        or not isinstance(root.body[-1], ConditionalChain)
        or not root.body[-1].body
        or not isinstance(root.body[-1].body[-1], (IFExpression, ElseIfExpression))
    ):
        msg = "An `elif_` expression may only follow an `if_` or an `elif_`"
        raise WorkflowError(msg)
    expr = ElseIfExpression(condition=condition)
    with expr.collect():
        yield
    root.body[-1].extend(expr)


@variable_tracker.track
@contextmanager
def else_() -> Generator[None, None, None]:
    """Workflow else statement.

    The equivalent of Python's else-statement.

    Example:
        ```python
        from laboneq import workflow


        @workflow.workflow
        def a_workflow(x):
            with workflow.if_(x == 1):
                ...
            with workflow.elif_(x == 2):
                ...
            with workflow.else_():
                ...
        ```

    Raises:
        WorkflowError: Expression is defined without `elif_()` or `if_()`
    """
    root = blocks.BlockBuilderContext.get_active()
    if not root:
        yield
        return
    if (
        not root.body
        or not isinstance(root.body[-1], ConditionalChain)
        or not root.body[-1].body
        or not isinstance(root.body[-1].body[-1], (IFExpression, ElseIfExpression))
    ):
        msg = "An `else_` expression may only follow an `if_` or an `elif_`"
        raise WorkflowError(msg)
    expr = ElseExpression()
    with expr.collect():
        yield
    root.body[-1].extend(expr)
