# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Block visitor."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.workflow.blocks import Block


class BlockVisitor:
    """A block tree visitor.

    This class is meant to be subclassed.

    Subclasses can implement custom visitor method for each block type.
    The format for naming the methods are `visit_<block type lowercase>`

        For example, to write a visitor method for `WorkflowBlock`:

            `def visit_workflowblock(self, block):`

    The visitor method should visit or call `.generic_visit()` to visit the
    children.
    """

    def __init__(self) -> None:
        pass

    def visit(self, block: Block) -> None:
        """Visit a block and its children recursively.

        Arguments:
            block: Root block.
        """
        visitor = getattr(
            self, f"visit_{block.__class__.__name__.lower()}", self.generic_visit
        )
        return visitor(block)

    def generic_visit(self, block: Block) -> None:
        """Visit block children.

        Arguments:
            block: Root block.
        """
        for child in block.iter_child_blocks():
            self.visit(child)
