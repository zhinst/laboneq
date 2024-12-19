# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.compiler.workflow.compiler_output import (
    CombinedRTCompilerOutputContainer,
)
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutputContainer,
)

from laboneq.compiler.seqc.linker import SeqCLinker
from laboneq.compiler.common.iface_linker import ILinker


_registered_hooks: dict[int, type[ILinker]] = {}


def register_linker_hook(device_class: int, hook: type[ILinker]):
    _registered_hooks[device_class] = hook


register_linker_hook(0, SeqCLinker)


def from_single_run(
    rt_compiler_output: RTCompilerOutputContainer, step_indices: list[int]
) -> CombinedRTCompilerOutputContainer:
    return CombinedRTCompilerOutputContainer(
        combined_output={
            device_class: _registered_hooks[device_class].combined_from_single_run(
                output, step_indices
            )
            for device_class, output in rt_compiler_output.codegen_output.items()
        },
        schedule=rt_compiler_output.schedule,
    )


def merge_compiler_runs(
    this: CombinedRTCompilerOutputContainer,
    new: RTCompilerOutputContainer,
    previous: RTCompilerOutputContainer,
    step_indices: list[int],
):
    for device_class, combined_output in this.combined_output.items():
        _registered_hooks[device_class].merge_combined_compiler_runs(
            combined_output,
            new.codegen_output[device_class],
            previous.codegen_output[device_class],
            step_indices,
        )


def repeat_previous(
    this: CombinedRTCompilerOutputContainer, previous: RTCompilerOutputContainer
):
    for device_class, combined_output in this.combined_output.items():
        _registered_hooks[device_class].repeat_previous(
            combined_output, previous.codegen_output[device_class]
        )
