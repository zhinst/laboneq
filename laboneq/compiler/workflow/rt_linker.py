# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from laboneq.compiler.code_generator import IntegrationTimes
from laboneq.compiler.code_generator.measurement_calculator import SignalDelays
from laboneq.compiler.code_generator.sampled_event_handler import FeedbackConnection
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.workflow.realtime_compiler import RealtimeCompilerOutput
from laboneq.core.exceptions import LabOneQException
from laboneq.data.scheduled_experiment import PulseMapEntry


def deep_compare(a: Any, b: Any) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, list):
        if len(a) != len(b):
            return False
        return all([deep_compare(_a, _b) for _a, _b in zip(a, b)])
    if isinstance(a, dict):
        if len(a) != len(b):
            return False
        if not deep_compare(list(a.keys()), list(b.keys())):
            return False
        return deep_compare(list(a.values()), list(b.values()))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b, equal_nan=True)
    return a == b


@dataclass
class RealtimeStep:
    device_id: str
    awg_id: int
    seqc_ref: str
    wave_indices_ref: str
    nt_step: list[int]


@dataclass
class CombinedRealtimeCompilerOutput:
    realtime_steps: list[RealtimeStep] = field(default_factory=list)
    command_table_match_offsets: dict[AwgKey, int] = field(default_factory=dict)
    feedback_connections: dict[str, FeedbackConnection] = field(default_factory=dict)
    feedback_registers: dict[AwgKey, int] = field(default_factory=dict)
    signal_delays: SignalDelays = field(default_factory=dict)
    integration_weights: dict = field(default_factory=dict)
    integration_times: IntegrationTimes = None
    simultaneous_acquires: dict[float, dict[str, str]] = field(default_factory=dict)
    total_execution_time: float = None
    max_execution_time_per_step: float = None
    src: list[dict[str, Any]] = field(default_factory=dict)
    waves: dict[str, dict[str, Any]] = field(default_factory=list)
    wave_indices: list[dict[str, Any]] = field(default_factory=dict)
    command_tables: list[dict[str, Any]] = field(default_factory=dict)
    pulse_map: dict[str, PulseMapEntry] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)


def make_seqc_name(awg: AwgKey, step_indices: list[int]) -> str:
    # Replace with UUID? Hash digest?
    step_indices_str = "[" + ",".join([str(i) for i in step_indices]) + "]"
    return f"seq_{awg.device_id}_{awg.awg_number}_{step_indices_str}.seqc"


def from_single_run(
    rt_compiler_output: RealtimeCompilerOutput, step_indices: list[int]
) -> CombinedRealtimeCompilerOutput:
    realtime_steps = []
    src = []
    command_tables = []
    wave_indices = []
    for awg, awg_src in rt_compiler_output.src.items():
        seqc_name = make_seqc_name(awg, step_indices)
        src.append({"filename": seqc_name, **awg_src})
        ct = rt_compiler_output.command_tables.get(awg)
        if ct is not None:
            command_tables.append({"seqc": seqc_name, **ct})

        wave_indices.append(
            {
                "filename": seqc_name,
                **rt_compiler_output.wave_indices[awg],
            }
        )
        realtime_steps.append(
            RealtimeStep(
                awg.device_id, awg.awg_number, seqc_name, seqc_name, step_indices
            )
        )
    return CombinedRealtimeCompilerOutput(
        realtime_steps=realtime_steps,
        command_table_match_offsets=rt_compiler_output.command_table_match_offsets,
        feedback_connections=rt_compiler_output.feedback_connections,
        feedback_registers=rt_compiler_output.feedback_registers,
        signal_delays=rt_compiler_output.signal_delays,
        integration_weights=rt_compiler_output.integration_weights,
        integration_times=rt_compiler_output.integration_times,
        simultaneous_acquires=rt_compiler_output.simultaneous_acquires,
        total_execution_time=rt_compiler_output.total_execution_time,
        max_execution_time_per_step=rt_compiler_output.total_execution_time,
        src=src,
        waves=rt_compiler_output.waves,
        command_tables=command_tables,
        wave_indices=wave_indices,
        pulse_map=rt_compiler_output.pulse_map,
        schedule=rt_compiler_output.schedule,
    )


def merge_compiler_runs(
    this: CombinedRealtimeCompilerOutput,
    new: RealtimeCompilerOutput,
    previous: RealtimeCompilerOutput,
    step_indices: list[int],
):
    if this.command_table_match_offsets != new.command_table_match_offsets:
        raise LabOneQException(
            "Command table match offsets do not match between real-time iterations"
        )
    if this.feedback_connections != new.feedback_connections:
        raise LabOneQException(
            "Feedback connections do not match between real-time iterations"
        )
    if this.feedback_registers != new.feedback_registers:
        raise LabOneQException(
            "Feedback registers do not match between real-time iterations"
        )
    if this.signal_delays != new.signal_delays:
        raise LabOneQException(
            "Signal delays do not match between real-time iterations"
        )
    if not deep_compare(this.integration_weights, new.integration_weights):
        # todo: this we probably want to allow in the future
        raise LabOneQException(
            "Integration weights do not match between real-time iterations"
        )
    if this.integration_times != new.integration_times:
        raise LabOneQException(
            "Integration times do not match between real-time iterations"
        )
    if this.simultaneous_acquires != new.simultaneous_acquires:
        raise LabOneQException(
            "Simultaneous acquires do not match between real-time iterations"
        )

    for awg, awg_src in new.src.items():
        seqc_name = make_seqc_name(awg, step_indices)

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

        if (
            previous_src == awg_src
            and previous_ct == new_ct
            and previous_wave_indices == new_wave_indices
            and deep_compare(previous_waves, new_waves)
        ):
            # No change in this iteration
            continue

            # todo: this can be more fine-grained. Maybe only the waveforms changed,
            #  but not the command table or the src.

        this.src.append({"filename": seqc_name, **awg_src})
        this.command_tables.append({"seqc": seqc_name, **new_ct})
        this.wave_indices.append({"filename": seqc_name, **new_wave_indices})
        this.waves.update(new_waves)
        this.realtime_steps.append(
            RealtimeStep(
                awg.device_id, awg.awg_number, seqc_name, seqc_name, step_indices
            )
        )
        this.max_execution_time_per_step = max(
            this.max_execution_time_per_step, new.total_execution_time
        )
