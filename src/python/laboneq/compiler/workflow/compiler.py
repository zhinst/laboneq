# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from laboneq.compiler.workflow import compiler_hooks
from laboneq.compiler.workflow.neartime_execution import (
    NtCompilerExecutor,
)

from . import compat

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_compiler_output import CombinedOutput
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

    combined_output: CombinedOutput
    schedule: compiler_rs.PulseSheetSchedule | None


def compile_whole_or_with_chunks(
    experiment: compiler_rs.ProcessedExperiment,
    execution: Statement,
    chunk_count: int | None,
    backend: str,
) -> CompiledOutput:
    """Compile the real-time part of every near-time step and link the results.

    This function is called from Rust Compiler in `compiler_module.compile_experiment()`
    """

    executor = NtCompilerExecutor(experiment, chunk_count, backend)
    executor.run(execution)

    combined = executor.combined_compiler_output()
    assert combined is not None, "Internal error: missing real-time compiler output"
    return CompiledOutput(
        combined_output=combined.combined_output, schedule=combined.schedule
    )
