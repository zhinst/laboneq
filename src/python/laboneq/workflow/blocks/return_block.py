# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow return block."""

from __future__ import annotations

from typing import Any, Callable

from laboneq.workflow import variable_tracker
from laboneq.workflow.blocks.block import Block, BlockBuilderContext
from laboneq.workflow.executor import ExecutionStatus, ExecutorState


class Namespace:
    """A class to provide attribute access to an immutable namespace.

    Arguments:
        **kwargs: Attributes to set to the namespace.
    """

    def __init__(self, **kwargs: object) -> None:
        self.__dict__.update(kwargs)

    def __setattr__(self, name, value):  # noqa: ANN001
        raise AttributeError("can't set attribute")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("can't delete attribute")

    def __repr__(self):
        attrs = ", ".join(f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({attrs})"

    def __eq__(self, other: object):
        if not isinstance(other, type(self)):
            return NotImplemented
        return vars(self) == vars(other)


class ReturnStatement(Block):
    """Return statement for a workflow.

    Sets the active workflow block output and interrupts the current workflow execution.

    Arguments:
        callback:
            Callback producing the value to be set for the workflow output.
        **kwargs:
            Keyword arguments for the callback.
    """

    def __init__(self, callback: Callable, /, **kwargs: object) -> None:
        self.callback = callback
        super().__init__(parameters=kwargs)

    @classmethod
    def from_value(cls, value: object) -> ReturnStatement:
        """Construct a return statement that returns the just given value.

        Arguments:
            value:
                The value to be returned as the workflow output.

        Returns:
            block:
                A return block.
        """
        return cls((lambda value: value), value=value)

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        if self.parameters:
            args = executor.resolve_inputs(self)
            executor.set_execution_output(self.callback(**args))
        executor.set_block_status(self, ExecutionStatus.FINISHED)
        executor.interrupt()

    def __str__(self):
        return "return_()"


@variable_tracker.track
def return_(output: Any | None = None, /, **kwargs: object) -> None:  # noqa: ANN401
    """Return statement of an workflow.

    Interrupts the current workflow execution and sets the active workflow output
    value either to the given `output` value or to a `Namespace` object by using
    the given keyword arguments.

    The equivalent of Python's `return` statement.

    Arguments:
        output: Value to be set for workflow output.
        **kwargs: Keyword arguments that are passed to the `Namespace` object.

    Raises:
        TypeError: Function is called with positional and keyword arguments.
    """
    if output is not None and kwargs:
        msg = "return_() takes either a single positional argument or keyword arguments"
        raise TypeError(msg)
    root = BlockBuilderContext.get_active()
    if root:
        if kwargs:
            root.extend(ReturnStatement(Namespace, **kwargs))
        else:
            root.extend(ReturnStatement.from_value(output))
