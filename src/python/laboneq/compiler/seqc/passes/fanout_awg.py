# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from functools import singledispatchmethod
import attrs
from typing import Iterable

from laboneq.compiler import ir as ir_mod
from laboneq.compiler.common import awg_info
from laboneq.compiler.seqc import ir as ir_seqc


class _SingleAWGTree:
    def __init__(self, awg: awg_info.AWGInfo):
        self._awg = awg
        self.signals = {x.id for x in awg.signals}
        self.stack: list[ir_mod.IntervalIR] = []

    def push_raw(self, start: int, node: ir_mod.IntervalIR):
        """Push a raw node that has no children."""
        if self.stack:
            self.stack[-1].children.append(node)
            self.stack[-1].children_start.append(start)

    def push_node(self, start: int, node: ir_mod.IntervalIR):
        """Push a new node into the AWG tree."""
        node = attrs.evolve(
            node,
            children=[],
            children_start=[],
            signals=node.signals & self.signals,
        )
        if self.stack:
            self.stack[-1].children.append(node)
            self.stack[-1].children_start.append(start)
        self.stack.append(node)

    def pop_node(self):
        """Mark last pushed node as finished."""
        if len(self.stack) != 1:
            self.stack.pop()

    def to_ir(self) -> ir_seqc.SingleAwgIR:
        # If the stack top is not root, the AWG is basically unused.
        if not self.stack or not isinstance(self.stack[0], ir_mod.RootScheduleIR):
            return ir_seqc.SingleAwgIR(length=0, awg=self._awg)
        root = self.stack[0]
        return ir_seqc.SingleAwgIR(
            awg=self._awg,
            children=root.children,
            children_start=root.children_start,
            signals=root.signals,
            length=root.length,
        )


class _AwgPruner:
    """IR pruner for AWGs.

    Walks the IR tree and splits it into subtrees for each awg.
    """

    # TODO(markush): Filter by device (Phase set, etc.)
    def __init__(self):
        self._awg_trees: list[_SingleAWGTree] = []
        self._signals: set[str] = set()

    def run(
        self, root: ir_mod.IntervalIR, start: int, awgs: list[awg_info.AWGInfo]
    ) -> list[ir_seqc.SingleAwgIR]:
        self._awg_trees = [_SingleAWGTree(awg) for awg in awgs]
        self._signals = set()
        for awg in self._awg_trees:
            self._signals.update(awg.signals)
        self.visit(root, start)
        res = [x.to_ir() for x in self._awg_trees]
        self._awg_trees = []
        self._signals = set()
        return res

    @singledispatchmethod
    def visit(self, node, start: int):
        raise RuntimeError("Nodes must be of type `IntervalIR`")

    @visit.register(ir_mod.InitialOscillatorFrequencyIR)
    @visit.register(ir_mod.SetOscillatorFrequencyIR)
    def _handle_oscillator_node(
        self,
        node: ir_mod.InitialOscillatorFrequencyIR | ir_mod.SetOscillatorFrequencyIR,
        start: int,
    ):
        # NOTE: Union on singledispatch available from Python 3.11
        # Oscillator IR does not expose signals within oscillators, so
        # we must go oscillator by oscillator.
        oscs_per_awg = [[]] * len(self._awg_trees)
        for osc in node.oscillators:
            for idx, intersect in [
                (idx, intersect)
                for idx, x in enumerate(self._awg_trees)
                if (intersect := osc.signals.intersection(x.signals))
            ]:
                oscs_per_awg[idx].append(attrs.evolve(osc, signals=intersect))
        for idx, oscs in enumerate(oscs_per_awg):
            self._awg_trees[idx].push_raw(start, attrs.evolve(node, oscillators=oscs))
        return None

    @visit.register
    def generic_visit(self, node: ir_mod.IntervalIR, start: int):
        if not node.signals.intersection(self._signals):
            return None
        awgs = [x for x in self._awg_trees if node.signals.intersection(x.signals)]
        for awg in awgs:
            awg.push_node(start, node)

        for start, child in node.iter_children():
            self.visit(child, start)

        for awg in awgs:
            awg.pop_node()


def fanout_awgs(tree: ir_mod.IRTree, awgs: Iterable[awg_info.AWGInfo]) -> ir_mod.IRTree:
    """Creates a new tree, where each top level interval represents a single AWG.

    Creates multiple `SingleAwgIR` instances at the top level of the tree schedule, each one
    wrapping a pruned version of the experiment containing only what is relevant to the
    individual AWGs.

    Does not modify the input.
    """
    children = []
    children_starts = []
    visitor = _AwgPruner()
    ir_awgs = visitor.run(tree.root, start=0, awgs=awgs)
    for awg in ir_awgs:
        children.append(awg)
        children_starts.append(0)
    return attrs.evolve(
        tree,
        root=attrs.evolve(
            tree.root,
            length=tree.root.length,
            children=children,
            children_start=children_starts,
        ),
    )
