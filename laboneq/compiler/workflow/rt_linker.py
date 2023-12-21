# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from functools import singledispatch

from typing import Any

import numpy as np

from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.workflow.compiler_output import (
    SeqCGenOutput,
    CombinedRTCompilerOutputContainer,
    CombinedRTOutputSeqC,
    CombinedRTOutputPrettyPrinter,
    PrettyPrinterOutput,
    RTCompilerOutputContainer,
    RTCompilerOutput,
    RealtimeStep,
    RealtimeStepPrettyPrinter,
)
from laboneq.core.exceptions.laboneq_exception import LabOneQException


def _deep_compare(a: Any, b: Any) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all([_deep_compare(_a, _b) for _a, _b in zip(a, b)])
    if isinstance(a, dict):
        if len(a) != len(b):
            return False
        if not _deep_compare(list(a.keys()), list(b.keys())):
            return False
        return _deep_compare(list(a.values()), list(b.values()))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    return a == b


def _make_seqc_name(awg: AwgKey, step_indices: list[int]) -> str:
    # Replace with UUID? Hash digest?
    step_indices_str = "[" + ",".join([str(i) for i in step_indices]) + "]"
    return f"seq_{awg.device_id}_{awg.awg_number}_{step_indices_str}.seqc"


@singledispatch
def _combined_from_single_run(output, step_indices: list[int]):
    raise NotImplementedError()


@singledispatch
def _merge_combined_compiler_runs(this, new, previous, step_indices: list[int]):
    raise NotImplementedError()


@_combined_from_single_run.register
def _(
    output: PrettyPrinterOutput, step_indices: list[int]
) -> CombinedRTOutputPrettyPrinter:
    step_indices_str = (
        "[" + ",".join([str(i) for i in step_indices]) + "]"
        if len(step_indices) > 0
        else "[0]"
    )
    key = f"pp_{step_indices_str}"
    return CombinedRTOutputPrettyPrinter(
        src={key: output.src},
        sections={key: output.sections},
        waves={key: output.waves},
    )


@_combined_from_single_run.register
def _(output: SeqCGenOutput, step_indices: list[int]) -> CombinedRTOutputSeqC:
    src = []
    command_tables = []
    wave_indices = []
    integration_weights = []
    for awg, awg_src in output.src.items():
        seqc_name = _make_seqc_name(awg, step_indices)
        src.append({"filename": seqc_name, **awg_src})
        ct = output.command_tables.get(awg)
        if ct is not None:
            command_tables.append({"seqc": seqc_name, **ct})

        wave_indices.append(
            {
                "filename": seqc_name,
                **output.wave_indices[awg],
            }
        )
        if awg in output.integration_weights:
            integration_weights.append(
                {"filename": seqc_name, "signals": output.integration_weights[awg]}
            )

    return CombinedRTOutputSeqC(
        feedback_connections=output.feedback_connections,
        signal_delays=output.signal_delays,
        integration_weights=integration_weights,
        integration_times=output.integration_times,
        simultaneous_acquires=output.simultaneous_acquires,
        total_execution_time=output.total_execution_time,
        max_execution_time_per_step=output.total_execution_time,
        src=src,
        waves=output.waves,
        command_tables=command_tables,
        wave_indices=wave_indices,
        pulse_map=output.pulse_map,
        feedback_register_configurations=output.feedback_register_configurations,
    )


def _check_compatibility(this, new):
    if this.feedback_connections != new.feedback_connections:
        raise LabOneQException(
            "Feedback connections do not match between real-time iterations"
        )
    if this.signal_delays != new.signal_delays:
        raise LabOneQException(
            "Signal delays do not match between real-time iterations"
        )
    if this.integration_times != new.integration_times:
        raise LabOneQException(
            "Integration times do not match between real-time iterations"
        )
    if this.simultaneous_acquires != new.simultaneous_acquires:
        raise LabOneQException(
            "Simultaneous acquires do not match between real-time iterations"
        )
    if this.feedback_register_configurations != new.feedback_register_configurations:
        raise LabOneQException(
            "Feedback register configurations do not match between real-time iterations"
        )


@_merge_combined_compiler_runs.register
def _(
    this: CombinedRTOutputSeqC,
    new: SeqCGenOutput,
    previous: SeqCGenOutput,
    step_indices: list[int],
):
    _check_compatibility(this, new)

    merged_ids = []

    for awg, awg_src in new.src.items():
        seqc_name = _make_seqc_name(awg, step_indices)

        previous_src = previous.src[awg]

        previous_ct = previous.command_tables.get(awg)
        new_ct = new.command_tables.get(awg)

        previous_wave_indices = previous.wave_indices.get(awg)
        new_wave_indices = new.wave_indices.get(awg)

        previous_waves = {
            name: wave
            for name, wave in previous.waves.items()
            if any(index_name in name for index_name in previous_wave_indices["value"])
        }
        new_waves = {
            name: wave
            for name, wave in new.waves.items()
            if any(index_name in name for index_name in new_wave_indices["value"])
        }

        previous_integration_weights = previous.integration_weights.get(awg)
        if previous_integration_weights is not None:
            previous_waves |= {
                name: wave
                for name, wave in previous.waves.items()
                if any(
                    index_name in name
                    for l in previous_integration_weights.values()
                    for index_name in l
                )
            }

        new_integration_weights = new.integration_weights.get(awg)
        if new_integration_weights is not None:
            new_waves |= {
                name: wave
                for name, wave in new.waves.items()
                if any(
                    index_name in name
                    for l in new_integration_weights.values()
                    for index_name in l
                )
            }

        if (
            previous_src == awg_src
            and previous_ct == new_ct
            and previous_wave_indices == new_wave_indices
            and _deep_compare(previous_waves, new_waves)
            and _deep_compare(previous_integration_weights, new_integration_weights)
        ):
            # No change in this iteration
            continue

            # todo: this can be more fine-grained. Maybe only the waveforms changed,
            #  but not the command table or the src.
        merged_ids.append((awg.device_id, awg.awg_number))
        this.src.append({"filename": seqc_name, **awg_src})
        if new_ct is not None:
            this.command_tables.append({"seqc": seqc_name, **new_ct})
        if new_wave_indices is not None:
            this.wave_indices.append({"filename": seqc_name, **new_wave_indices})
        if new_integration_weights is not None:
            this.integration_weights.append(
                {"filename": seqc_name, "signals": new_integration_weights}
            )
        this.waves.update(new_waves)
        this.max_execution_time_per_step = max(
            this.max_execution_time_per_step, new.total_execution_time
        )

    return merged_ids


@_merge_combined_compiler_runs.register
def _(
    this: CombinedRTOutputPrettyPrinter,
    new: PrettyPrinterOutput,
    previous: PrettyPrinterOutput,
    step_indices: list[int],
):
    return []


def make_real_time_step_from_container(
    rt_compiler_output: RTCompilerOutputContainer, step_indices: list[int]
):
    return [
        step
        for rtoutput in rt_compiler_output.codegen_output.values()
        for step in make_real_time_step(rtoutput, step_indices)
    ]


def make_real_time_step(rt_compiler_output: RTCompilerOutput, step_indices: list[int]):
    if isinstance(rt_compiler_output, SeqCGenOutput):
        realtime_steps = []
        for awg in rt_compiler_output.src.keys():
            seqc_name = _make_seqc_name(awg, step_indices)
            realtime_steps.append(
                RealtimeStep(
                    device_id=awg.device_id,
                    awg_id=awg.awg_number,
                    seqc_ref=seqc_name,
                    wave_indices_ref=seqc_name,
                    kernel_indices_ref=seqc_name,
                    nt_step=step_indices,
                )
            )
    elif isinstance(rt_compiler_output, PrettyPrinterOutput):
        realtime_steps = [RealtimeStepPrettyPrinter("pp", 0, step_indices)]
    else:
        raise TypeError
    return realtime_steps


def from_single_run(
    rt_compiler_output: RTCompilerOutputContainer, step_indices: list[int]
) -> CombinedRTCompilerOutputContainer:
    return CombinedRTCompilerOutputContainer(
        combined_output={
            device_class: _combined_from_single_run(output, step_indices)
            for device_class, output in rt_compiler_output.codegen_output.items()
        },
        realtime_steps=make_real_time_step_from_container(
            rt_compiler_output, step_indices
        ),
        schedule=rt_compiler_output.schedule,
    )


def merge_compiler_runs(
    this: CombinedRTCompilerOutputContainer,
    new: RTCompilerOutputContainer,
    previous: RTCompilerOutputContainer,
    step_indices: list[int],
):
    for device_class, combined_output in this.combined_output.items():
        merged_ids = _merge_combined_compiler_runs(
            combined_output,
            new.codegen_output[device_class],
            previous.codegen_output[device_class],
            step_indices,
        )
        new_realtime_steps = make_real_time_step_from_container(new, step_indices)
        for new_realtime_step in new_realtime_steps:
            if (
                new_realtime_step.device_id,
                new_realtime_step.awg_id,
            ) in merged_ids:
                this.realtime_steps.append(new_realtime_step)
