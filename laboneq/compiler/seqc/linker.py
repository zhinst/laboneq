# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np

from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.seqc.measurement_calculator import (
    SignalDelays,
    IntegrationTimes,
)
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.compiler.common.iface_linker import ILinker
from laboneq.compiler.common.iface_compiler_output import (
    RTCompilerOutput,
    CombinedOutput,
    RealtimeStepBase,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.data.scheduled_experiment import (
    PulseMapEntry,
    CompilerArtifact,
    ArtifactsCodegen,
)


@dataclass
class CombinedRTOutputSeqC(CombinedOutput):
    feedback_connections: dict[str, FeedbackConnection] = field(default_factory=dict)
    signal_delays: SignalDelays = field(default_factory=dict)
    integration_weights: list[dict[str, Any]] = field(default_factory=list)
    integration_times: IntegrationTimes = None
    simultaneous_acquires: list[dict[str, str]] = field(default_factory=list)
    src: list[dict[str, Any]] = field(default_factory=dict)
    waves: dict[str, dict[str, Any]] = field(default_factory=list)
    wave_indices: list[dict[str, Any]] = field(default_factory=dict)
    command_tables: list[dict[str, Any]] = field(default_factory=dict)
    pulse_map: dict[str, PulseMapEntry] = field(default_factory=dict)
    feedback_register_configurations: dict[AwgKey, FeedbackRegisterConfig] = field(
        default_factory=dict
    )
    realtime_steps: list[RealtimeStep] = field(default_factory=list)

    total_execution_time: float = 0
    max_execution_time_per_step: float = 0

    def get_artifacts(self) -> CompilerArtifact:
        return ArtifactsCodegen(
            src=self.src,
            waves=list(self.waves.values()),
            wave_indices=self.wave_indices,
            command_tables=self.command_tables,
            pulse_map=self.pulse_map,
            integration_weights=self.integration_weights,
        )


@dataclass
class SeqCGenOutput(RTCompilerOutput):
    feedback_connections: Dict[str, FeedbackConnection]
    signal_delays: SignalDelays
    integration_weights: dict[AwgKey, dict[str, list[str]]]
    integration_times: IntegrationTimes
    simultaneous_acquires: list[Dict[str, str]]
    src: Dict[AwgKey, Dict[str, Any]]
    waves: Dict[str, Dict[str, Any]]
    wave_indices: Dict[AwgKey, Dict[str, Any]]
    command_tables: Dict[AwgKey, Dict[str, Any]]
    pulse_map: Dict[str, PulseMapEntry]
    feedback_register_configurations: Dict[AwgKey, FeedbackRegisterConfig]

    total_execution_time: float = 0


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


class SeqCLinker(ILinker):
    @staticmethod
    def combined_from_single_run(output, step_indices: list[int]):
        src = []
        command_tables = []
        wave_indices = []
        integration_weights = []
        for awg, awg_src in output.src.items():
            awg: AwgKey
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
            realtime_steps=SeqCLinker.make_realtime_step(output, step_indices),
        )

    @staticmethod
    def merge_combined_compiler_runs(
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
                if any(
                    index_name in name for index_name in previous_wave_indices["value"]
                )
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

            merged_ids.append(awg)

            for pulse_id, entry in new.pulse_map.items():
                if pulse_id not in this.pulse_map:
                    this.pulse_map[pulse_id] = entry
                else:
                    this.pulse_map[pulse_id].waveforms.update(entry.waveforms)

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

        for new_realtime_step in SeqCLinker.make_realtime_step(new, step_indices):
            if (
                AwgKey(new_realtime_step.device_id, new_realtime_step.awg_id)
                in merged_ids
            ):
                this.realtime_steps.append(new_realtime_step)

    @staticmethod
    def make_realtime_step(rt_compiler_output: SeqCGenOutput, step_indices: list[int]):
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
        return realtime_steps


def _make_seqc_name(awg: AwgKey, step_indices: list[int]) -> str:
    # Replace with UUID? Hash digest?
    step_indices_str = "[" + ",".join([str(i) for i in step_indices]) + "]"
    return f"seq_{awg.device_id}_{awg.awg_number}_{step_indices_str}.seqc"


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


@dataclass
class RealtimeStep(RealtimeStepBase):
    device_id: str
    awg_id: int
    seqc_ref: str
    wave_indices_ref: str
    kernel_indices_ref: str
    nt_step: list[int]
