# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterator, Any

from laboneq.compiler.common.device_type import DeviceType
from laboneq.core.exceptions import LabOneQException
from laboneq.compiler import ir


def inline_sections(node: ir.IntervalIR, start: int | None = None) -> None:
    """Inline sections that have no useful information.

    The children of the removed sections are moved upwards
    and the relative timings are adjusted.

    This pass operates in-place.
    """
    start = 0 if start is None else 0
    # We iterate backwards as the node.children is modified
    for i in range(len(node.children) - 1, -1, -1):
        child = node.children[i]
        start_child = node.children_start[i]
        inline_sections(child, start + start_child)
        # Cannot use `isinstance()`, it must be the exact type `ir.SectionIR`
        if type(child) is ir.SectionIR:
            if not child.prng_setup and not child.trigger_output:
                node.children.pop(i)
                node.children_start.pop(i)
                node.children.extend(child.children)
                node.children_start.extend(
                    [x + start_child for x in child.children_start]
                )
    idxs = sorted(range(len(node.children_start)), key=lambda i: node.children_start[i])
    node.children_start = [node.children_start[i] for i in idxs]
    node.children = [node.children[i] for i in idxs]


class _SectionInliner:
    def __init__(self, ir: ir.IRTree):
        self._ir = ir
        self._signals = {signal.uid: signal for signal in self._ir.signals}

    def run(self):
        [(top_start, top)] = self._visit(0, node=self._ir.root, case_section=None)
        assert top_start == 0
        assert isinstance(top, ir.RootScheduleIR)

    def _visit(
        self, start: int, node, case_section: str | None
    ) -> Iterator[tuple[int, Any]]:
        if not isinstance(node, ir.IntervalIR):
            yield start, node
            return

        in_branch = case_section is not None

        if in_branch:
            if isinstance(node, ir.PulseIR):
                if node.increment_oscillator_phase or node.set_oscillator_phase:
                    for signal_id in node.signals:
                        signal = self._signals[signal_id]

                        if (
                            osc := signal.oscillator
                        ) is not None and not signal.oscillator.is_hardware:
                            raise LabOneQException(
                                f"Conditional 'increment_oscillator_phase' or"
                                f" 'set_oscillator_phase' of software oscillator"
                                f" '{osc.uid}' on signal '{signal_id}' not supported"
                            )
                        assert node.set_oscillator_phase is None, (
                            "cannot set HW osc phase"
                        )

                        dt = DeviceType.from_device_info_type(signal.device.device_type)
                        if dt.is_qa_device:
                            # The _actual_ problem is that UHFQA and SHFQA do not support CT
                            # phase registers. In practice, they don't because such a feature
                            # is irrelevant on a QA.
                            raise LabOneQException(
                                f"Conditional 'increment_oscillator_phase' of signal"
                                f"'{signal_id}' not supported on device type '{dt.name}'"
                            )

        if isinstance(node, ir.CaseIR):
            assert case_section is None, "Cannot nest case() sections"
            case_section = node.section

        # recurse on the children
        new_children = zip(
            *(
                (new_child_start, new_child)
                for child_start, child in node.iter_children()
                for (new_child_start, new_child) in self._visit(
                    child_start, child, case_section
                )
            )
        )
        node.children_start = list(next(new_children, []))
        node.children = list(next(new_children, []))

        if in_branch:
            if type(node) is ir.SectionIR:
                # dissolve the node, yield its children instead
                for child_start, child in node.iter_children():
                    if isinstance(child, ir.PulseIR):
                        child.section = case_section
                    yield start + child_start, child
                return
            else:
                msg = "No special sections permitted inside case() blocks"
                assert not isinstance(node, ir.SectionIR) or isinstance(
                    node, ir.CaseIR
                ), msg
        yield start, node


def inline_sections_in_branch(ir: ir.IRTree):
    """Replace non-control flow sections by their contents.

    This pass operates in-place."""
    _SectionInliner(ir).run()
