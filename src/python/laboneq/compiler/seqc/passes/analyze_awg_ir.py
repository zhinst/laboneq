# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from laboneq.compiler import ir
from dataclasses import dataclass


@dataclass
class AwgCompileInfo:
    has_readout_feedback: bool = False


def _collect_compile_info(node: ir.IntervalIR, info: AwgCompileInfo):
    if type(node) is ir.MatchIR and node.handle is not None:
        info.has_readout_feedback = True
        return
    for _, child in node.iter_children():
        _collect_compile_info(child, info)
        if info.has_readout_feedback:
            break


def analyze_awg_ir(root: ir.IntervalIR) -> AwgCompileInfo:
    """Analyzes AWG IR and collects relevant information from it."""
    info = AwgCompileInfo()
    _collect_compile_info(root, info)
    return info
