# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import logging

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.common.signal_obj import SignalObj
from laboneq.compiler.ir.interval_ir import IntervalIR
from laboneq.compiler.ir.ir import IR
from laboneq.compiler.ir.pulse_ir import PulseIR
from laboneq.compiler.ir.section_ir import SectionIR
from laboneq.compiler.workflow.compiler_output import PrettyPrinterOutput

_logger = logging.getLogger(__name__)


class PrettyPrinter:
    def _print_outline_rec(self, ir: IntervalIR, nesting: int):
        self._src += f"{'  '*nesting}{ir.__class__.__name__}\n"
        for c in ir.children:
            self._print_outline_rec(c, nesting=nesting + 1)

    def _collect_sections(self, ir: IntervalIR, sections: list[str]):
        if isinstance(ir, SectionIR):
            sections.append(ir.section)
        for c in ir.children:
            self._collect_sections(c, sections)

    def _collect_waves(self, ir: IntervalIR, waves: list[str]):
        if isinstance(ir, PulseIR):
            waves.append(ir.pulse.pulse.uid)
        for c in ir.children:
            self._collect_waves(c, waves)

    def __init__(self, ir: IR = None, settings: CompilerSettings = None):
        self._ir = ir
        self._settings = settings

    def generate_code(self, signal_objs: list[SignalObj]):
        self._src = ""
        self._waves = []
        self._sections = []

        self._print_outline_rec(self._ir.root, 0)
        self._collect_sections(self._ir.root, self._sections)
        self._collect_waves(self._ir.root, self._waves)

    def fill_output(self):
        return PrettyPrinterOutput(
            src=self._src, waves=self._waves, sections=self._sections
        )
