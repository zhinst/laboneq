# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import laboneq.compiler.workflow.reporter  # noqa: F401

# reporter import is required to register the CompilationReportGenerator hook
from laboneq.compiler.workflow import compiler_hooks
from laboneq.compiler.workflow.compiler_hooks import (
    CompiledOutputParams,
    get_compiler_hooks,
)
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
)

from . import compat

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common import compiler_settings
    from laboneq.data.scheduled_experiment import CompilerArtifact
    from laboneq.executor.executor import Statement

_logger = logging.getLogger(__name__)


def compile_capnp(
    capnp_bytes: bytes,
    backend: str,
    compiler_settings: dict | None = None,
) -> bytes:
    """Compile the given capnp data which represents an experiment and device setup."""
    _logger.info("Starting LabOne Q Compiler run...")
    compiler_module = compiler_hooks.resolve_compiler_module(backend)
    payload = compiler_module.compile_experiment(
        capnp_bytes,
        packed=compat.use_packed_capnp(),
        compiler_settings=compat.sanitize_compiler_settings(compiler_settings or {}),
    )
    _logger.info("Finished LabOne Q Compiler run.")
    return payload


@dataclass
class CompiledOutput:
    """Result shape that follows the field defined in `CompilationOutputPy` in Rust."""

    artifacts: CompilerArtifact
    schedule: compiler_rs.PulseSheetSchedule | None


def compile_whole_or_with_chunks(
    experiment: compiler_rs.ProcessedExperiment,
    execution: Statement,
    chunk_count: int | None,
    backend: str,
    compiler_settings: compiler_settings.CompilerSettings,
) -> CompiledOutput:
    """Compile the given experiment.

    This function is called from Rust Compiler in `compiler_module.compile_experiment()`
    """

    executor = NtCompilerExecutor(experiment, compiler_settings, chunk_count, backend)
    executor.run(execution)

    combined_compiler_output = executor.combined_compiler_output()
    assert combined_compiler_output is not None, (
        "Internal error: missing real-time compiler output"
    )

    executor.finalize()

    combined_output = combined_compiler_output.combined_output

    artifacts = get_compiler_hooks(backend).compiled_output(
        CompiledOutputParams(combined_compiler_output=combined_output)
    )

    return CompiledOutput(
        artifacts=artifacts,
        schedule=combined_compiler_output.schedule,
    )
