# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A workflow tree graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from laboneq.workflow import blocks


class _BlockTreeDisplay(blocks.BlockVisitor):
    """Tree display for workflow blocks."""

    @dataclass
    class Context:
        lines: list[str] = field(default_factory=list)
        branch: str = "├─"
        depth_changes: list[bool] = field(default_factory=list)

    def _format(self, prefix: str, block: blocks.Block) -> str:
        if prefix:
            prefix = prefix + " "
        formatted = str(block).replace("\n", "")
        return f"{prefix}{formatted}"

    def visit(self, block: blocks.Block, ctx: Context | None = None) -> list[str]:
        """Visit the node and display a tree graph.

        Does not display hidden blocks.
        """
        if ctx is None:
            ctx = self.Context()
        if ctx.depth_changes:
            prefix = "".join("│  " if p else "   " for p in ctx.depth_changes[:-1])
            prefix += ctx.branch
        else:
            prefix = ""
        ctx.lines.append(self._format(prefix, block))
        ctx.depth_changes.append(True)
        visible_body = [x for x in block.body if not x.hidden]
        for child in visible_body[:-1]:
            ctx.branch = "├─"
            self.visit(child, ctx)
        if visible_body:
            ctx.branch = "└─"
            ctx.depth_changes[-1] = False
            self.visit(visible_body[-1], ctx)
        ctx.depth_changes.pop()
        return ctx.lines


class WorkflowGraphTree:
    """Workflow graph as a tree."""

    def __init__(self, root: blocks.WorkflowBlock) -> None:
        self._root = root

    def __str__(self):
        displ = _BlockTreeDisplay().visit(self._root)
        return "\n".join(displ)

    def _repr_pretty_(self, pp, cycle):  # noqa: ANN001, ARG002, ANN202
        # For Notebooks
        pp.text(str(self))
