# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The QCCS backend's compiler hooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq._rust import BACKEND_NAME
from laboneq.compiler.workflow.compiler_hooks import (
    CompilerHooks,
    SetupDescriptionResult,
)

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs
    from laboneq.compiler.common.iface_linker import ILinker
    from laboneq.compiler.workflow.compiler_hooks import CompiledOutputParams
    from laboneq.data.scheduled_experiment import CompilerArtifact
    from laboneq.dsl.device.device_setup import DeviceSetup


class CompilerHooksSeqC(CompilerHooks):
    @staticmethod
    def backend_name() -> str:
        return BACKEND_NAME

    @staticmethod
    def compiler_module() -> compiler_rs:
        from laboneq._rust import compiler

        return compiler

    @staticmethod
    def linker() -> type[ILinker]:
        from laboneq.compiler.seqc.linker import SeqCLinker

        return SeqCLinker

    @staticmethod
    def compiled_output(args: CompiledOutputParams) -> CompilerArtifact:
        from laboneq.compiler.seqc.artifact_generator import populate_codegen_artifacts

        return populate_codegen_artifacts(args.combined_compiler_output)

    @staticmethod
    def build_setup_description(
        device_setup: DeviceSetup, signal_map: dict[str, str]
    ) -> SetupDescriptionResult:
        from laboneq.qccs.compiler import setup_description

        return SetupDescriptionResult(
            setup_description=setup_description.build(device_setup),
            signal_map=signal_map,
        )
