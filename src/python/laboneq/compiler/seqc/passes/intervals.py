# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections import defaultdict
import attrs
from laboneq.compiler import ir


@attrs.define(repr=False, slots=True)
class EmptyInterval:
    # The signal interval is acting
    signal: str
    # Start in samples
    start: int
    # End in samples
    end: int
    # State of the match case
    state: int


class _EmptyCaseIntervals:
    """Create interval events for empty match cases.

    This is temporary. Once we have migrated the code generator to directly consume the
    IR (rather than the event list) for generating the interval tree, this will likely be
    rendered obsolete.
    """

    def __init__(self):
        self._empty_branch_intervals: list[EmptyInterval] = []

    def run(self, node: ir.IntervalIR, start: int = 0) -> list[EmptyInterval]:
        self.visit(node, start)
        result = self._empty_branch_intervals
        self._empty_branch_intervals = defaultdict(list)
        return result

    def visit(self, node: ir.IntervalIR, start: int) -> None:
        visitor = getattr(self, f"visit_{node.__class__.__name__}", self.generic_visit)
        return visitor(node, start)

    def generic_visit(self, node: ir.IntervalIR, start: int) -> None:
        for start_ch, child in node.iter_children():
            self.visit(child, start + start_ch)

    def visit_CaseIR(self, node: ir.CaseIR, start: int) -> None:
        if node.children or node.length == 0:
            return
        for signal in node.signals:
            self._empty_branch_intervals.append(
                EmptyInterval(
                    signal=signal,
                    start=start,
                    end=start + node.length,
                    state=node.state,
                )
            )


def collect_empty_intervals(root: ir.IntervalIR, start: int = 0) -> list[EmptyInterval]:
    """Collect empty match case intervals."""
    return _EmptyCaseIntervals().run(root, start)
