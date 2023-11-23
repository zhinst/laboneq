# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict


from laboneq.compiler.code_generator.measurement_calculator import (
    IntegrationTimes,
    SignalDelays,
)
from laboneq.compiler.code_generator.sampled_event_handler import FeedbackConnection
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.common.feedback_register_config import FeedbackRegisterConfig
from laboneq.data.scheduled_experiment import (
    ArtifactsCodegen,
    ArtifactsPrettyPrinter,
    CompilerArtifact,
    PulseMapEntry,
)


def _make_seqc_name(awg: AwgKey, step_indices: list[int]) -> str:
    # Replace with UUID? Hash digest?
    step_indices_str = "[" + ",".join([str(i) for i in step_indices]) + "]"
    return f"seq_{awg.device_id}_{awg.awg_number}_{step_indices_str}.seqc"


@dataclass
class RealtimeStep:
    device_id: str
    awg_id: int
    seqc_ref: str
    wave_indices_ref: str
    nt_step: list[int]


@dataclass
class RealtimeStepPrettyPrinter:
    device_id: str
    awg_id: int
    nt_step: list[int]


class RTCompilerOutput:
    pass


class CombinedOutput:
    pass


@dataclass
class CombinedRealtimeCompilerOutputCode(CombinedOutput):
    feedback_connections: dict[str, FeedbackConnection] = field(default_factory=dict)
    signal_delays: SignalDelays = field(default_factory=dict)
    integration_weights: dict = field(default_factory=dict)
    integration_times: IntegrationTimes = None
    simultaneous_acquires: dict[float, dict[str, str]] = field(default_factory=dict)
    src: list[dict[str, Any]] = field(default_factory=dict)
    waves: dict[str, dict[str, Any]] = field(default_factory=list)
    wave_indices: list[dict[str, Any]] = field(default_factory=dict)
    command_tables: list[dict[str, Any]] = field(default_factory=dict)
    pulse_map: dict[str, PulseMapEntry] = field(default_factory=dict)
    feedback_register_configurations: dict[AwgKey, FeedbackRegisterConfig] = field(
        default_factory=dict
    )

    total_execution_time: float = 0
    max_execution_time_per_step: float = 0

    def get_artifacts(self) -> CompilerArtifact:
        return ArtifactsCodegen(
            src=self.src,
            waves=list(self.waves.values()),
            wave_indices=self.wave_indices,
            command_tables=self.command_tables,
            pulse_map=self.pulse_map,
        )


@dataclass
class CombinedRealtimeCompilerOutputPrettyPrinter(CombinedOutput):
    src: list[dict[str, Any]] = field(default_factory=dict)
    waves: list[dict[str, list[str]]] = field(default_factory=dict)
    sections: list[dict[str, list[str]]] = field(default_factory=dict)

    total_execution_time: float = 0
    max_execution_time_per_step: float = 0

    feedback_register_configurations: dict[AwgKey, FeedbackRegisterConfig] = field(
        default_factory=dict
    )

    def get_artifacts(self) -> CompilerArtifact:
        return ArtifactsPrettyPrinter(
            src=self.src, waves=self.waves, sections=self.sections
        )


@dataclass
class PrettyPrinterOutput(RTCompilerOutput):
    src: str
    sections: list[str]
    waves: list[str]

    total_execution_time: float = 0

    def real_time_steps(self, step_indices) -> list[RealtimeStep]:
        return [RealtimeStepPrettyPrinter("pp", 0, step_indices)]


@dataclass
class CodegenOutput(RTCompilerOutput):
    feedback_connections: Dict[str, FeedbackConnection]
    signal_delays: SignalDelays
    integration_weights: Dict
    integration_times: IntegrationTimes
    simultaneous_acquires: Dict[float, Dict[str, str]]
    src: Dict[AwgKey, Dict[str, Any]]
    waves: Dict[str, Dict[str, Any]]
    wave_indices: Dict[AwgKey, Dict[str, Any]]
    command_tables: Dict[AwgKey, Dict[str, Any]]
    pulse_map: Dict[str, PulseMapEntry]
    feedback_register_configurations: Dict[AwgKey, FeedbackRegisterConfig]

    total_execution_time: float = 0

    def real_time_steps(self, step_indices) -> list[RealtimeStep]:
        realtime_steps = []
        for awg in self.src.keys():
            seqc_name = _make_seqc_name(awg, step_indices)
            realtime_steps.append(
                RealtimeStep(
                    awg.device_id, awg.awg_number, seqc_name, seqc_name, step_indices
                )
            )
        return realtime_steps


@dataclass()
class RealtimeCompilerOutput:
    codegen_output: dict[int, RTCompilerOutput]
    schedule: dict[str, Any]

    def realtime_steps(self, step_indices: list[int]):
        return [
            step
            for rtoutput in self.codegen_output.values()
            for step in rtoutput.real_time_steps(step_indices)
        ]


@dataclass
class CombinedRealtimeCompilerOutput:
    combined_output: dict[int, CombinedOutput]
    # NOTE(mr) does this make sense? do we want realtime steps "globally" or stored per device class?
    realtime_steps: dict[int, list[RealtimeStep]] = field(default_factory=dict)
    schedule: dict[str, Any] = field(default_factory=dict)

    def get_artifacts(self):
        if len(self.combined_output) == 1:
            return list(self.combined_output.values())[0].get_artifacts()
        else:
            return {
                device_class: combined_output.get_artifacts()
                for device_class, combined_output in self.combined_output.items()
            }

    def get_feedback_register_configurations(self, key: AwgKey):
        for output in self.combined_output.values():
            if key in output.feedback_register_configurations:
                return output.feedback_register_configurations[key]

    def add_total_execution_time(self, other):
        for device_class, combined_output in self.combined_output.items():
            combined_output.total_execution_time += other.codegen_output[
                device_class
            ].total_execution_time

    @property
    def total_execution_time(self):
        return max([c.total_execution_time for c in self.combined_output.values()])

    @property
    def max_execution_time_per_step(self):
        return max(
            [c.max_execution_time_per_step for c in self.combined_output.values()]
        )
