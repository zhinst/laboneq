# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutputContainer,
)
from laboneq.compiler.workflow.compiler_hooks import get_compiler_hooks

if TYPE_CHECKING:
    from laboneq._rust import compiler as compiler_rs

    from .parameter_store import ParameterStore


def compile_realtime(
    experiment: compiler_rs.ProcessedExperiment,
    near_time_parameters: ParameterStore[str, float],
    device_class: int,
) -> RTCompilerOutputContainer:
    compiler_module: compiler_rs = get_compiler_hooks(device_class).compiler_module()
    result = compiler_module.compile_realtime(
        experiment=experiment,
        parameters={
            k: v
            for k, v in near_time_parameters.items()
            if k not in ("__chunk_index", "__chunk_count")
        },
        chunking_info=None
        if "__chunk_index" not in near_time_parameters
        else (
            near_time_parameters["__chunk_index"],
            near_time_parameters["__chunk_count"],
        ),
    )
    # Flush used `nt_parameters` so that they get registered
    for used_parameter in result.used_parameters:
        near_time_parameters.mark_used(used_parameter)

    pulse_sheet_schedule = (
        _prepare_pulse_sheet_schedule(result.pulse_sheet_schedule)
        if result.pulse_sheet_schedule is not None
        else None
    )

    return RTCompilerOutputContainer(
        device_class=device_class,
        codegen_output=result.codegen_output(),
        schedule=pulse_sheet_schedule,
    )


def _prepare_pulse_sheet_schedule(schedule: compiler_rs.PulseSheetSchedule) -> Schedule:
    return Schedule(
        event_list=schedule["event_list"],
        event_list_truncated=schedule["event_list_truncated"],
        section_info=schedule["section_info"],
        section_signals_with_children=schedule["section_signals_with_children"],
        sampling_rates=schedule["sampling_rates"],
    )


class Schedule(TypedDict):
    event_list: list[dict]
    event_list_truncated: bool
    section_info: dict[str, dict]
    section_signals_with_children: dict[str, list[str]]
    sampling_rates: list[tuple[list[str], float]]
