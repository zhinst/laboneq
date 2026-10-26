# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.compiler.workflow.compiler_hooks import get_compiler_hooks
from laboneq.compiler.workflow.compiler_output import (
    CombinedRTCompilerOutputContainer,
)

if TYPE_CHECKING:
    from laboneq.compiler.common.iface_compiler_output import (
        RTCompilerOutputContainer,
    )


def from_single_run(
    rt_compiler_output: RTCompilerOutputContainer, step_indices: list[int]
) -> CombinedRTCompilerOutputContainer:
    combined_output = (
        get_compiler_hooks(rt_compiler_output.backend)
        .linker()
        .combined_from_single_run(rt_compiler_output.codegen_output, step_indices)
    )
    return CombinedRTCompilerOutputContainer(
        backend=rt_compiler_output.backend,
        combined_output=combined_output,
        schedule=rt_compiler_output.schedule,
    )


def merge_compiler_runs(
    this: CombinedRTCompilerOutputContainer,
    new: RTCompilerOutputContainer,
    previous: RTCompilerOutputContainer,
    step_indices: list[int],
):
    get_compiler_hooks(this.backend).linker().merge_combined_compiler_runs(
        this.combined_output, new.codegen_output, previous.codegen_output, step_indices
    )


def repeat_run(
    this: CombinedRTCompilerOutputContainer,
    run: RTCompilerOutputContainer,
    step_indices: list[int],
):
    get_compiler_hooks(this.backend).linker().repeat_run(
        this.combined_output, run.codegen_output, step_indices
    )
