# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from laboneq.compiler.workflow import compiler_hooks

from . import compat

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
