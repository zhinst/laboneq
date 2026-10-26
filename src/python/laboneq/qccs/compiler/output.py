# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Finishes the QCCS backend's compilation attempts and produces its artifacts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.compiler.common.compiler_settings import CompilerSettings
from laboneq.compiler.seqc.artifact_generator import populate_codegen_artifacts
from laboneq.core.utilities import seqc_compile
from laboneq.core.utilities.seqc_compile import SeqCCompileItem
from laboneq.qccs.compiler import reporter

if TYPE_CHECKING:
    from laboneq.compiler.seqc.linker import CombinedRTOutputSeqC
    from laboneq.data.scheduled_experiment import CompilerArtifact


def finish_attempt(
    combined: CombinedRTOutputSeqC,
    *,
    log_report: bool,
    ignore_resource_limitation_errors: bool,
    uses_chunking: bool,
):
    """Compiles the SeqC programs of `combined` in place.

    Logs the compilation report first when `log_report` is set.

    Raises `ResourceLimitationError` when a program exceeds a device resource,
    unless `ignore_resource_limitation_errors` is set.
    """
    if log_report:
        reporter.log_report(combined)
    settings = CompilerSettings(
        IGNORE_RESOURCE_LIMITATION_ERRORS=ignore_resource_limitation_errors
    )
    _compile_seqc(combined, settings, uses_chunking)


def produce_output(combined: CombinedRTOutputSeqC) -> CompilerArtifact:
    return populate_codegen_artifacts(combined).get_artifacts()


def _compile_seqc(
    combined: CombinedRTOutputSeqC, settings: CompilerSettings, uses_chunking: bool
):
    seqc_items = [
        SeqCCompileItem(
            dev_type=seqc_program.dev_type,
            dev_opts=seqc_program.dev_opts,
            awg_index=seqc_program.awg_index,
            sequencer=seqc_program.sequencer,
            sampling_rate=seqc_program.sampling_rate,
            code=seqc_program.src,
            filename=seqc_program.filename,
            pipeliner=uses_chunking,
        )
        for seqc_program in combined.src.values()
        if seqc_program.dev_type is not None
    ]
    seqc_compile.awg_compile(seqc_items, settings)
    for seqc_item in seqc_items:
        combined.src[seqc_item.filename].elf = seqc_item.elf
