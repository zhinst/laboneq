# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A module for defining visitors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.workflow.blocks import BlockVisitor, WorkflowBlock

if TYPE_CHECKING:
    from collections.abc import Sequence

    from laboneq.workflow.blocks.block import Block


class SpecificBlockTypeCollector(BlockVisitor):
    """Collector for specific type of blocks."""

    def __init__(self, root: WorkflowBlock) -> None:
        self.root = root
        self._block_types = []
        self._blocks = []

    def collect(self, block_types: Sequence[type[Block]]) -> list[Block]:
        """Collect specific type of blocks.

        Arguments:
            block_types: A sequence of block types to be collected.

        Returns:
            A list of found blocks.
        """
        self._block_types = block_types
        try:
            self.visit(self.root)
        finally:
            blocks = self._blocks
            self._blocks = []
        return blocks

    def generic_visit(self, block: Block) -> None:
        """Visit the thing."""
        if isinstance(block, tuple(self._block_types)):
            self._blocks.append(block)
        super().generic_visit(block)
