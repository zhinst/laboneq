# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.compiler.common.iface_compiler_output import NeartimeStepBase
from laboneq.core.exceptions import LabOneQException
from laboneq.data.artifacts_qccs import (
    COMPLEX_USAGE,
    ArtifactsCodegen,
    CommandTableMapEntry,
    ParameterPhaseIncrementMap,
)
from laboneq.data.awg_info import AwgKey
from laboneq.data.nt_step_key import NtStepKey

if TYPE_CHECKING:
    import laboneq._rust.codegenerator as codegen_rs
    from laboneq.compiler.common.integration_times import IntegrationTimes
    from laboneq.core.types.enums.wave_type import WaveType
    from laboneq.data.artifacts_qccs import (
        AcquireLength,
        AwgWeights,
        CodegenWaveform,
        Initialization,
        IntegratorAllocation,
        OscillatorParam,
        PulseMapEntry,
        RealtimeExecutionInit,
        ResultSource,
    )
    from laboneq.data.scheduled_experiment import CompilerArtifact


@dataclass
class CombinedRTOutputSeqC:
    integration_times: IntegrationTimes
    awg_properties: dict[AwgKey, codegen_rs.AwgProperties] = field(default_factory=dict)
    # key - SeqC name
    integration_weights: dict[str, AwgWeights] = field(default_factory=dict)
    result_handle_maps: dict[ResultSource, list[set[str]]] = field(default_factory=dict)
    src: dict[str, SeqCProgram] = field(default_factory=dict)
    waves: dict[str, CodegenWaveform] = field(default_factory=dict)
    requires_long_readout: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    wave_indices: list[dict[str, str | dict[str, tuple[int, WaveType]]]] = field(
        default_factory=list
    )
    command_tables: list[dict[str, Any]] = field(default_factory=list)
    pulse_map: dict[str, PulseMapEntry] = field(default_factory=dict)
    parameter_phase_increment_map: dict[str, ParameterPhaseIncrementMap] = field(
        default_factory=dict
    )
    feedback_register_configurations: dict[
        AwgKey, codegen_rs.FeedbackRegisterConfig
    ] = field(default_factory=dict)
    neartime_steps: list[NeartimeStep] = field(default_factory=list)
    measurements: list[codegen_rs.Measurement] = field(default_factory=list)
    _total_execution_time: float = 0
    _max_execution_time_per_step: float = 0
    # SeqC ref of the program each AWG runs after the latest linked step
    _loaded_seqc_refs: dict[AwgKey, str] = field(default_factory=dict)
    integration_unit_allocations: dict[
        AwgKey, list[codegen_rs.IntegrationUnitAllocation]
    ] = field(default_factory=dict)
    channel_properties: dict[AwgKey, list[codegen_rs.ChannelProperties]] = field(
        default_factory=dict
    )
    device_properties: list[codegen_rs.DeviceProperties] = field(default_factory=list)
    ppc_settings: list[codegen_rs.PpcSettings] = field(default_factory=list)
    result_lengths: dict[AwgKey, int] = field(default_factory=dict)
    initializations: list[Initialization] = field(default_factory=list)
    realtime_execution_init: list[RealtimeExecutionInit] = field(default_factory=list)
    oscillator_params: list[OscillatorParam] = field(default_factory=list)
    integrator_allocations: list[IntegratorAllocation] = field(default_factory=list)
    acquire_lengths: list[AcquireLength] = field(default_factory=list)

    def get_artifacts(self) -> CompilerArtifact:
        src: list[dict[str, str | bytes]] = []
        for seqc in self.src.values():
            seqc_entry: dict[str, str | bytes] = {
                "filename": seqc.filename,
                "text": seqc.src,
            }
            if isinstance(seqc.elf, bytes):
                seqc_entry["elf"] = seqc.elf
            src.append(seqc_entry)
        return ArtifactsCodegen(
            src=src,
            waves=self.waves,
            requires_long_readout=self.requires_long_readout,
            wave_indices=self.wave_indices,
            command_tables=self.command_tables,
            pulse_map=self.pulse_map,
            integration_weights=self.integration_weights,
            parameter_phase_increment_map=self.parameter_phase_increment_map,
            result_handle_maps=self.result_handle_maps,
            initializations=self.initializations,
            realtime_execution_init=self.realtime_execution_init,
            oscillator_params=self.oscillator_params,
            integrator_allocations=self.integrator_allocations,
            acquire_lengths=self.acquire_lengths,
        )

    def get_raw_acquire_length(
        self,
        signal_id: str,
    ) -> int:
        integration_info = self.integration_times.signal_infos.get(signal_id)
        assert integration_info is not None
        return integration_info.length_in_samples

    @property
    def total_execution_time(self) -> float:
        return self._total_execution_time

    @property
    def max_execution_time_per_step(self) -> float:
        return self._max_execution_time_per_step


@dataclass
class SeqCProgram:
    src: str
    dev_type: str | None = None
    dev_opts: list[str] = field(default_factory=list)
    awg_index: int = 0
    sequencer: str = "auto"
    sampling_rate: float | None = None
    filename: str = ""
    elf: bytes | None = None


@dataclass
class SeqCGenOutput:
    integration_weights: dict[AwgKey, AwgWeights]
    integration_times: IntegrationTimes
    result_handle_maps: dict[ResultSource, list[set[str]]]
    src: dict[AwgKey, SeqCProgram]
    waves: dict[str, CodegenWaveform]
    requires_long_readout: dict[str, list[str]]
    wave_indices: dict[AwgKey, dict[str, tuple[int, WaveType]]]
    command_tables: dict[AwgKey, dict[str, Any]]
    pulse_map: dict[str, PulseMapEntry]
    parameter_phase_increment_map: dict[AwgKey, dict[str, list]]
    feedback_register_configurations: dict[AwgKey, codegen_rs.FeedbackRegisterConfig]
    total_execution_time: float = 0
    measurements: list[codegen_rs.Measurement] = field(default_factory=list)
    integration_unit_allocations: dict[
        AwgKey, list[codegen_rs.IntegrationUnitAllocation]
    ] = field(default_factory=dict)
    channel_properties: dict[AwgKey, list[codegen_rs.ChannelProperties]] = field(
        default_factory=dict
    )
    awg_properties: dict[AwgKey, codegen_rs.AwgProperties] = field(default_factory=dict)
    device_properties: list[codegen_rs.DeviceProperties] = field(default_factory=list)
    ppc_settings: list[codegen_rs.PpcSettings] = field(default_factory=list)
    # Each AWG must have the same result lengths across real-time executions.
    # This means that for near-time steps and chunked experiments,
    # there should only be one entry per AWG
    result_lengths: dict[AwgKey, int] = field(default_factory=dict)


def _check_compatibility(this: CombinedRTOutputSeqC, new: SeqCGenOutput):
    def scheduler_delays(
        channels: dict[AwgKey, list[codegen_rs.ChannelProperties]],
    ) -> dict[str, float]:
        return {
            c.signal: c.scheduler_delay
            for channels in channels.values()
            for c in channels
        }

    def precompensations(
        channels: dict[AwgKey, list[codegen_rs.ChannelProperties]],
    ) -> dict[str, dict | None]:
        return {
            c.signal: c.precompensation
            for channels in channels.values()
            for c in channels
        }

    if scheduler_delays(this.channel_properties) != scheduler_delays(
        new.channel_properties
    ):
        raise LabOneQException(
            "Signal scheduler delays do not match between real-time iterations"
        )
    if precompensations(this.channel_properties) != precompensations(
        new.channel_properties
    ):
        raise LabOneQException(
            "Signal precompensations do not match between real-time iterations"
        )
    if this.result_lengths != new.result_lengths:
        raise LabOneQException("Measurements do not match between real-time iterations")
    if this.result_handle_maps != new.result_handle_maps:
        raise LabOneQException("Measurements do not match between real-time iterations")
    if this.integration_times != new.integration_times:
        raise LabOneQException("Measurements do not match between real-time iterations")
    if this.feedback_register_configurations != new.feedback_register_configurations:
        raise LabOneQException(
            "Feedback register configurations do not match between real-time iterations"
        )
    if this.ppc_settings != new.ppc_settings:
        raise LabOneQException("PPC settings do not match between real-time iterations")
    if this.integration_unit_allocations != new.integration_unit_allocations:
        raise LabOneQException(
            "Integration unit allocations do not match between real-time iterations"
        )
    if this.device_properties != new.device_properties:
        raise LabOneQException(
            "Device properties do not match between real-time iterations"
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


class SeqCLinker:
    @staticmethod
    def combined_from_single_run(output: SeqCGenOutput, step_indices: list[int]):
        src = {}
        command_tables = []
        wave_indices: list[dict[str, str | dict[str, tuple[int, WaveType]]]] = []
        integration_weights: dict[str, AwgWeights] = {}
        parameter_phase_increment_map: dict[str, ParameterPhaseIncrementMap] = {}
        for awg, seqc_program in output.src.items():
            seqc_name = _make_seqc_name(awg, step_indices)
            seqc_program.filename = seqc_name
            src[seqc_name] = seqc_program
            ct = output.command_tables.get(awg)
            if ct is not None:
                command_tables.append({"seqc": seqc_name, **ct})

            wave_indices.append(
                {
                    "filename": seqc_name,
                    "value": output.wave_indices[awg],
                }
            )
            if awg in output.integration_weights:
                integration_weights[seqc_name] = output.integration_weights[awg]
            _extend_parameter_phase_increment_map(
                parameter_phase_increment_map, output, seqc_name, awg_key=awg
            )

        return CombinedRTOutputSeqC(
            awg_properties=output.awg_properties,
            integration_times=output.integration_times,
            integration_weights=integration_weights,
            result_handle_maps=output.result_handle_maps,
            _total_execution_time=output.total_execution_time,
            _max_execution_time_per_step=output.total_execution_time,
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
            _loaded_seqc_refs=_seqc_refs(output),
            parameter_phase_increment_map=parameter_phase_increment_map,
            measurements=output.measurements,
            integration_unit_allocations=output.integration_unit_allocations,
            channel_properties=output.channel_properties,
            ppc_settings=output.ppc_settings,
            device_properties=output.device_properties,
            result_lengths=output.result_lengths,
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

        for awg, seqc_program in new.src.items():
            seqc_name = _make_seqc_name(awg, step_indices)

            previous_src = previous.src[awg].src

            previous_ct = previous.command_tables.get(awg)
            new_ct = new.command_tables.get(awg)

            previous_wave_indices = previous.wave_indices.get(awg, {})
            new_wave_indices = new.wave_indices.get(awg, {})

            previous_waves: dict[str, CodegenWaveform] = {
                name: wave
                for name, wave in previous.waves.items()
                if any(index_name in name for index_name in previous_wave_indices)
            }
            new_waves: dict[str, CodegenWaveform] = {
                name: wave
                for name, wave in new.waves.items()
                if any(index_name in name for index_name in new_wave_indices)
            }

            previous_integration_weights = previous.integration_weights.get(awg)
            if previous_integration_weights is not None:
                previous_waves |= {
                    name: wave
                    for name, wave in previous.waves.items()
                    if any(
                        index_name.id in name
                        for index_name in previous_integration_weights
                    )
                }

            new_integration_weights = new.integration_weights.get(awg)
            if new_integration_weights is not None:
                new_waves |= {
                    name: wave
                    for name, wave in new.waves.items()
                    if any(
                        index_name.id in name for index_name in new_integration_weights
                    )
                }

            if (
                previous_src == seqc_program.src
                and previous_ct == new_ct
                and previous_wave_indices == new_wave_indices
                and _deep_compare(previous_waves, new_waves)
                and _deep_compare(previous_integration_weights, new_integration_weights)
            ):
                # No change in this iteration
                seqc_program.filename = previous.src[awg].filename
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

            seqc_program.filename = seqc_name
            this.src[seqc_name] = seqc_program
            if new_ct is not None:
                this.command_tables.append({"seqc": seqc_name, **new_ct})
            if new_wave_indices is not None:
                this.wave_indices.append(
                    {"filename": seqc_name, "value": new_wave_indices}
                )
            if new_integration_weights is not None:
                this.integration_weights[seqc_name] = new_integration_weights
            this.waves.update(new_waves)

        this._max_execution_time_per_step = max(
            this._max_execution_time_per_step, new.total_execution_time
        )
        this._total_execution_time += new.total_execution_time

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
        this._loaded_seqc_refs = _seqc_refs(new)

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
    def repeat_run(
        this: CombinedRTOutputSeqC,
        run: SeqCGenOutput,
        step_indices: list[int],
    ):
        for awg, seqc_program in run.src.items():
            if this._loaded_seqc_refs[awg] == seqc_program.filename:
                continue
            assert isinstance(awg.awg_id, int)
            this.neartime_steps.append(
                NeartimeStep(
                    device_id=awg.device_id,
                    awg_id=awg.awg_id,
                    seqc_ref=seqc_program.filename,
                    wave_indices_ref=seqc_program.filename,
                    kernel_indices_ref=seqc_program.filename,
                    key=NtStepKey(tuple(step_indices)),
                )
            )
        this._loaded_seqc_refs = _seqc_refs(run)
        this._total_execution_time += run.total_execution_time


def _seqc_refs(output: SeqCGenOutput) -> dict[AwgKey, str]:
    return {awg: seqc_program.filename for awg, seqc_program in output.src.items()}


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
        if a.keys() != b.keys():
            return False
        return all(_deep_compare(a[k], b[k]) for k in a)
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
