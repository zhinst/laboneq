# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow for block."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Callable, TypeVar, cast

from laboneq.workflow import variable_tracker
from laboneq.workflow.blocks.block import Block
from laboneq.workflow.executor import ExecutionStatus, ExecutorState
from laboneq.workflow.reference import Reference

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


class ForExpression(Block):
    """For expression.

    A block that iterates workflow blocks over the given values.

    The expression will always fully execute regardless if the workflow
    is partially executed or not.

    Arguments:
        values: An iterable.
            Iterable can contain workflow objects.
        loop_indexer: A callable to produce dynamic loop indexes for each item.
            The callable must accept a single argument, the item currently being
            iterated.
            The index should be a hashable and unique across the items in the loop.
            Usually they are simple Python types such as
            `str`, `int` or tuples of these.

            When `loop_indexer` is `None`, each task executed within the loop is given
            a loop index from `0..len(values) - 1`.
    """

    def __init__(
        self,
        values: Iterable | Reference,
        loop_indexer: Callable[[object], object] | None = None,
    ) -> None:
        super().__init__(parameters={"values": values})
        self._ref = Reference(self)
        self._loop_indexer = loop_indexer or None

    @property
    def ref(self) -> Reference:
        """Reference to the object."""
        return self._ref

    def __enter__(self) -> Reference:
        """Enter the loop context.

        Returns:
            Individual values of the given iterable.
        """
        super().__enter__()
        return self.ref

    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""
        vals = executor.resolve_inputs(self)["values"]
        executor.set_block_status(self, ExecutionStatus.IN_PROGRESS)
        # Disable run until within the loop
        # TODO: Add support if seen necessary
        run_until = executor.settings.run_until
        executor.settings.run_until = None
        try:
            for idx, val in enumerate(vals):
                executor.set_variable(self.ref, val)
                with executor.scoped_index(
                    self._loop_indexer(val) if self._loop_indexer else idx
                ):
                    for block in self.body:
                        block.execute(executor)
        finally:
            executor.settings.run_until = run_until
        executor.set_block_status(self, ExecutionStatus.FINISHED)

    def __str__(self):
        return "for_()"


T = TypeVar("T")


@variable_tracker.track
@contextmanager
def for_(
    values: Iterable[T], loop_indexer: Callable[[object], str] | None = None
) -> Generator[T, None, None]:
    """For expression to iterate over the values within a code block.

    The equivalent of Python's for loop.

    Arguments:
        values: An iterable.
        loop_indexer: A callable to produce dynamic loop indexes for each item.
            The callable must accept a single argument, the item currently being
            iterated.
            The index should be a hashable and unique across the items in the loop.
            Usually they are simple Python types such as
            `str`, `int` or tuples of these.

            When `loop_indexer` is `None`, each task executed within the loop is given
            a loop index from `0..len(values) - 1`.

    Example:
        ```python
        with workflow.for_([1, 2, 3]) as x:
            ...
        ```
    """
    with ForExpression(values=values, loop_indexer=loop_indexer) as x:
        yield cast(T, x)
