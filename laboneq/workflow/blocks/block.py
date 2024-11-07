# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Workflow block base class."""

from __future__ import annotations

import abc
from contextlib import contextmanager
from typing import TYPE_CHECKING

from laboneq.workflow._context import LocalContext

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from laboneq.workflow.executor import ExecutorState


class Block(abc.ABC):
    """A base class for workflow blocks.

    A block can be an individual task or a collection of other blocks.

    Classes inheriting from `Block` must define the following methods:

        - `execute()`: A method that executes the block and it's children defined
            in `Block.body`.

    Arguments:
        parameters: Expected input parameters of the block.
    """

    def __init__(self, parameters: dict | None = None) -> None:
        self._parameters = parameters or {}
        self._body: list[Block] = []

    @property
    def parameters(self) -> dict:
        """Input parameters of the block."""
        return self._parameters

    @property
    def name(self) -> str:
        """Name of the block."""
        return self.__class__.__name__

    @property
    def hidden(self) -> bool:
        """Whether or not the block is a hidden block.

        Hidden blocks are generally used only for execution and
        they are not deemed relevant in results.
        """
        return False

    @property
    def body(self) -> list[Block]:
        """Body of the block.

        A list of other blocks that are defined within this block.
        """
        return self._body

    def extend(self, blocks: Block | Iterable[Block]) -> None:
        """Extend the body of the block."""
        if isinstance(blocks, Block):
            self._body.append(blocks)
        else:
            self._body.extend(blocks)

    def iter_child_blocks(self) -> Generator[Block]:
        """Iterate over the children of this block."""
        yield from self.body

    @contextmanager
    def collect(self) -> Generator[None, None, None]:
        """Collect blocks defined within the context.

        Every block defined within the context are added to the body of the
        block.
        """
        self.__enter__()
        yield
        BlockBuilderContext.exit()

    def __enter__(self):
        BlockBuilderContext.enter(self)

    def __exit__(self, exc_type, exc_value, traceback):  # noqa: ANN001
        BlockBuilderContext.exit()
        active_ctx = BlockBuilderContext.get_active()
        if active_ctx:
            active_ctx.extend(self)

    @abc.abstractmethod
    def execute(self, executor: ExecutorState) -> None:
        """Execute the block."""


class BlockBuilderContext(LocalContext[Block]):
    """Workflow block builder context."""

    _scope = "block_builder"
