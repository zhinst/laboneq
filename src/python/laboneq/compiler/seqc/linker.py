# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from laboneq.compiler.common.feedback_connection import FeedbackConnection
from laboneq.compiler.common.shfppc_sweeper_config import SHFPPCSweeperConfig
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
    NeartimeStepBase,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.data.recipe import NtStepKey
from laboneq.data.scheduled_experiment import (
    AwgWeights,
    CodegenWaveform,
    PulseMapEntry,
    CompilerArtifact,
    ArtifactsCodegen,
    ParameterPhaseIncrementMap,
    COMPLEX_USAGE,
    CommandTableMapEntry,
)


@dataclass
class CombinedRTOutputSeqC(CombinedOutput):
    feedback_connections: dict[str, FeedbackConnection] = field(default_factory=dict)
    signal_delays: SignalDelays = field(default_factory=dict)
    # key - SeqC name
    integration_weights: dict[str, AwgWeights] = field(default_factory=dict)
    integration_times: IntegrationTimes | None = None
    simultaneous_acquires: list[dict[str, str]] = field(default_factory=list)
    src: list[dict[str, Any]] = field(default_factory=list)
    waves: dict[str, CodegenWaveform] = field(default_factory=dict)
    requires_long_readout: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    wave_indices: list[dict[str, Any]] = field(default_factory=list)
    command_tables: list[dict[str, Any]] = field(default_factory=list)
    pulse_map: dict[str, PulseMapEntry] = field(default_factory=dict)
    parameter_phase_increment_map: dict[str, ParameterPhaseIncrementMap] = field(
        default_factory=dict
    )
    feedback_register_configurations: dict[AwgKey, FeedbackRegisterConfig] = field(
        default_factory=dict
    )
    neartime_steps: list[NeartimeStep] = field(default_factory=list)
    shfppc_sweep_configurations: dict[AwgKey, SHFPPCSweeperConfig] = field(
        default_factory=dict
    )

    total_execution_time: float = 0
    max_execution_time_per_step: float = 0

    def get_artifacts(self) -> CompilerArtifact:
        return ArtifactsCodegen(
            src=self.src,
            waves=self.waves,
            requires_long_readout=self.requires_long_readout,
            wave_indices=self.wave_indices,
            command_tables=self.command_tables,
            pulse_map=self.pulse_map,
            integration_weights=self.integration_weights,
            parameter_phase_increment_map=self.parameter_phase_increment_map,
        )


@dataclass
class SeqCGenOutput(RTCompilerOutput):
    feedback_connections: dict[str, FeedbackConnection]
    signal_delays: SignalDelays
    integration_weights: dict[AwgKey, AwgWeights]
    integration_times: IntegrationTimes
    simultaneous_acquires: list[dict[str, str]]
    src: dict[AwgKey, dict[str, Any]]
    waves: dict[str, CodegenWaveform]
    requires_long_readout: dict[str, list[str]]
    wave_indices: dict[AwgKey, dict[str, Any]]
    command_tables: dict[AwgKey, dict[str, Any]]
    pulse_map: dict[str, PulseMapEntry]
    parameter_phase_increment_map: dict[AwgKey, dict[str, list]]
    feedback_register_configurations: dict[AwgKey, FeedbackRegisterConfig]
    shfppc_sweep_configurations: dict[AwgKey, SHFPPCSweeperConfig]

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
    if this.shfppc_sweep_configurations != new.shfppc_sweep_configurations:
        raise LabOneQException(
            "SHFPPC sweep configurations do not match between real-time iterations"
        )


def _extend_parameter_phase_increment_map(
    ppim: dict[str, ParameterPhaseIncrementMap],
    new: SeqCGenOutput,
    ct_ref: str,
    awg_key: AwgKey,
):
    try:
        parameter_phase_increment_map = (
            new.parameter_phase_increment_map[  # @IgnoreException
                awg_key
            ]
        )
    except KeyError:
        return

    for param_name, targets in parameter_phase_increment_map.items():
        this_param_targets = ppim.setdefault(param_name, ParameterPhaseIncrementMap())
        for target in targets:
            entry = (
                CommandTableMapEntry(ct_ref, target)
                if target != COMPLEX_USAGE
                else target
            )
            this_param_targets.entries.append(entry)


class SeqCLinker(ILinker):
    @staticmethod
    def combined_from_single_run(output: SeqCGenOutput, step_indices: list[int]):
        src = []
        command_tables = []
        wave_indices = []
        integration_weights: dict[str, AwgWeights] = {}
        parameter_phase_increment_map: dict[str, ParameterPhaseIncrementMap] = {}
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
                integration_weights[seqc_name] = output.integration_weights[awg]
            _extend_parameter_phase_increment_map(
                parameter_phase_increment_map, output, seqc_name, awg_key=awg
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
            requires_long_readout=defaultdict(list, output.requires_long_readout),
            command_tables=command_tables,
            wave_indices=wave_indices,
            pulse_map=output.pulse_map,
            feedback_register_configurations=output.feedback_register_configurations,
            neartime_steps=SeqCLinker.make_neartime_execution_step(
                output, step_indices
            ),
            parameter_phase_increment_map=parameter_phase_increment_map,
            shfppc_sweep_configurations=output.shfppc_sweep_configurations,
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

            previous_waves: dict[str, CodegenWaveform] = {
                name: wave
                for name, wave in previous.waves.items()
                if any(
                    index_name in name for index_name in previous_wave_indices["value"]
                )
            }
            new_waves: dict[str, CodegenWaveform] = {
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
                        index_name.id in name
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
                        index_name.id in name
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

            _extend_parameter_phase_increment_map(
                this.parameter_phase_increment_map, new, seqc_name, awg
            )

            this.src.append({"filename": seqc_name, **awg_src})
            if new_ct is not None:
                this.command_tables.append({"seqc": seqc_name, **new_ct})
            if new_wave_indices is not None:
                this.wave_indices.append({"filename": seqc_name, **new_wave_indices})
            if new_integration_weights is not None:
                this.integration_weights[seqc_name] = new_integration_weights
            this.waves.update(new_waves)
            this.max_execution_time_per_step = max(
                this.max_execution_time_per_step, new.total_execution_time
            )
            this.total_execution_time += new.total_execution_time

        for (
            new_device_id,
            new_requires_long_readout,
        ) in new.requires_long_readout.items():
            dev_requires_long_readout = this.requires_long_readout[new_device_id]
            for signal_id in new_requires_long_readout:
                if signal_id not in dev_requires_long_readout:
                    dev_requires_long_readout.append(signal_id)

        for new_neartime_execution_step in SeqCLinker.make_neartime_execution_step(
            new, step_indices
        ):
            if (
                AwgKey(
                    new_neartime_execution_step.device_id,
                    new_neartime_execution_step.awg_id,
                )
                in merged_ids
            ):
                this.neartime_steps.append(new_neartime_execution_step)

    @staticmethod
    def make_neartime_execution_step(
        rt_compiler_output: SeqCGenOutput, step_indices: list[int]
    ):
        neartime_execution_steps = []
        for awg in rt_compiler_output.src.keys():
            seqc_name = _make_seqc_name(awg, step_indices)
            assert isinstance(awg.awg_id, int)
            neartime_execution_steps.append(
                NeartimeStep(
                    device_id=awg.device_id,
                    awg_id=awg.awg_id,
                    seqc_ref=seqc_name,
                    wave_indices_ref=seqc_name,
                    kernel_indices_ref=seqc_name,
                    key=NtStepKey(tuple(step_indices)),
                )
            )
        return neartime_execution_steps

    @staticmethod
    def repeat_previous(this: CombinedRTOutputSeqC, previous: SeqCGenOutput):
        this.total_execution_time += previous.total_execution_time


def _make_seqc_name(awg: AwgKey, step_indices: list[int]) -> str:
    # Replace with UUID? Hash digest?
    step_indices_str = "[" + ",".join([str(i) for i in step_indices]) + "]"
    return f"seq_{awg.device_id}_{awg.awg_id}_{step_indices_str}.seqc"


def _deep_compare(a: Any, b: Any) -> bool:
    if type(a) is not type(b):
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
class NeartimeStep(NeartimeStepBase):
    device_id: str
    awg_id: int
    seqc_ref: str
    wave_indices_ref: str
    kernel_indices_ref: str
    key: NtStepKey
