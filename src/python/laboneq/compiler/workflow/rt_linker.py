# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.compiler import CompilerSettings
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutputContainer,
)
from laboneq.compiler.workflow.compiler_hooks import get_compiler_hooks
from laboneq.compiler.workflow.compiler_output import (
    CombinedRTCompilerOutputContainer,
)


def from_single_run(
    rt_compiler_output: RTCompilerOutputContainer, step_indices: list[int]
) -> CombinedRTCompilerOutputContainer:
    combined_output = (
        get_compiler_hooks(rt_compiler_output.device_class)
        .linker()
        .combined_from_single_run(rt_compiler_output.codegen_output, step_indices)
    )
    return CombinedRTCompilerOutputContainer(
        device_class=rt_compiler_output.device_class,
        combined_output=combined_output,
        schedule=rt_compiler_output.schedule,
    )


def merge_compiler_runs(
    this: CombinedRTCompilerOutputContainer,
    new: RTCompilerOutputContainer,
    previous: RTCompilerOutputContainer,
    step_indices: list[int],
):
    get_compiler_hooks(this.device_class).linker().merge_combined_compiler_runs(
        this.combined_output, new.codegen_output, previous.codegen_output, step_indices
    )


def repeat_previous(
    this: CombinedRTCompilerOutputContainer, previous: RTCompilerOutputContainer
):
    get_compiler_hooks(this.device_class).linker().repeat_previous(
        this.combined_output, previous.codegen_output
    )


def finalize(this: CombinedRTCompilerOutputContainer, settings: CompilerSettings):
    get_compiler_hooks(this.device_class).linker().finalize(
        this.combined_output, settings
    )
