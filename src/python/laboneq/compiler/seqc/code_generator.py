# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import defaultdict
from itertools import groupby
from typing import TYPE_CHECKING

import numpy as np
import orjson

from laboneq.compiler.common.integration_times import (
    IntegrationTimes,
    SignalIntegrationInfo,
)
from laboneq.compiler.seqc.linker import SeqCGenOutput, SeqCProgram
from laboneq.core.types.enums.wave_type import WaveType
from laboneq.data.awg_info import AwgKey
from laboneq.data.scheduled_experiment import (
    COMPLEX_USAGE,
    CodegenWaveform,
    PulseMapEntry,
    ResultSource,
    WeightInfo,
)

from .types import SignalDelay

if TYPE_CHECKING:
    from laboneq._rust import codegenerator as codegen_rs

# This is used as a workaround for the SHFQA requiring that for sampled pulses,  abs(s)  < 1.0 must hold
# to be able to play pulses with an amplitude of 1.0, we scale complex pulses by this factor
SHFQA_COMPLEX_SAMPLE_SCALING = 1 - 1e-10


def generate_output(output: codegen_rs.SeqCGenOutput) -> SeqCGenOutput:
    integration_unit_allocations = {}
    awg_properties = {}
    channel_properties = defaultdict(list)
    signal_delays = {}
    integration_times = {}
    result_lengths = {}
    integration_kernels = {}
    integration_weights = {}
    sampled_waveforms = {}
    src = {}
    feedback_register_config = {}
    wave_indices_all = {}
    command_tables = {}
    parameter_phase_increment_map = {}

    waves = {}
    requires_long_readout = defaultdict(list)
    pulse_map = {}

    for awg_output in output.awg_results:
        awg_key = AwgKey(*awg_output.awg_properties.key)
        integration_unit_allocations[awg_key] = awg_output.integration_unit_allocations

        awg_properties[awg_key] = awg_output.awg_properties
        channel_properties[awg_key].extend(awg_output.channel_properties)
        for signal, delay in awg_output.signal_delays.items():
            signal_delays[signal] = SignalDelay(on_device=delay)
        for signal, integration_time in awg_output.integration_lengths.items():
            integration_times[signal] = SignalIntegrationInfo(
                is_play=integration_time.is_play,
                length_in_samples=integration_time.length,
            )
        if awg_output.result_length is not None:
            result_lengths[awg_key] = awg_output.result_length
        integration_kernels[awg_key] = awg_output.integration_kernels
        integration_weights[awg_key] = [
            WeightInfo(
                id=iw.basename,
                integration_units=iw.integration_units,
                downsampling_factor=iw.downsampling_factor,
            )
            for iw in awg_output.integration_weights
        ]

        sampled_waveforms[awg_key] = awg_output.sampled_waveforms
        seqc_program = awg_output.seqc
        src[awg_key] = SeqCProgram(
            src=seqc_program.src,
            dev_type=seqc_program.dev_type,
            dev_opts=seqc_program.dev_opts,
            awg_index=seqc_program.awg_index,
            sequencer=seqc_program.sequencer,
            sampling_rate=seqc_program.sampling_rate,
        )
        feedback_register_config[awg_key] = awg_output.feedback_register_config
        wave_indices_all[awg_key] = {
            k: (filename, WaveType(wavetype))
            for k, (filename, wavetype) in awg_output.wave_indices
        }
        if awg_output.command_table:
            ct = orjson.loads(awg_output.command_table)
            command_tables[awg_key] = {"ct": ct}
            parameter_phase_increment_map[awg_key] = {
                k: [i if i >= 0 else COMPLEX_USAGE for i in v]
                for k, v in awg_output.parameter_phase_increment_map.items()
            }

        _generate_waves(
            output=output,
            awg_output=awg_output,
            waves=waves,
            pulse_map=pulse_map,
            requires_long_readout=requires_long_readout,
        )

    # check that there are no duplicate filenames in the wave pool (HBAR-1079)
    _waves = sorted(
        [(filename, wave.samples) for filename, wave in waves.items()],
        key=lambda w: w[0],
    )
    for _, group in groupby(_waves, key=lambda w: w[0]):
        group = list(group)
        assert all(np.all(group[0][1] == g[1]) for g in group[1:])

    return SeqCGenOutput(
        awg_properties=awg_properties,
        signal_delays=signal_delays,
        integration_weights=integration_weights,
        integration_times=IntegrationTimes(signal_infos=integration_times),
        result_handle_maps={
            ResultSource(k.device_id, k.awg_id, k.integrator_idx): [
                set(item) for item in v
            ]
            for k, v in output.result_handle_maps.items()
        },
        result_lengths=result_lengths,
        src=src,
        total_execution_time=output.total_execution_time,
        waves=waves,
        requires_long_readout=requires_long_readout,
        wave_indices=wave_indices_all,
        command_tables=command_tables,
        pulse_map=pulse_map,
        parameter_phase_increment_map=parameter_phase_increment_map,
        feedback_register_configurations=feedback_register_config,
        measurements=output.measurements,
        integration_unit_allocations=integration_unit_allocations,
        channel_properties=channel_properties,
        ppc_settings=output.ppc_settings,
        device_properties=output.device_properties,
    )


def _generate_waves(
    output: codegen_rs.SeqCGenOutput,
    awg_output: codegen_rs.AwgCodeGenerationResult,
    waves: dict[str, CodegenWaveform],
    pulse_map: dict[str, PulseMapEntry],
    requires_long_readout: dict[str, list[str]],
):
    awg_key = AwgKey(*awg_output.awg_properties.key)
    device_type: str = next(
        device.device_type
        for device in output.device_properties
        if device.uid == awg_key.device_id
    )
    awg_properties = awg_output.awg_properties
    integration_weights = awg_output.integration_kernels
    sampled_waveforms = awg_output.sampled_waveforms

    # Handle integration weights separately
    # Group by signal
    weights_by_signal = defaultdict(list)
    for weight in integration_weights:
        for signal in weight.signals:
            weights_by_signal[signal].append(weight)
    for signal, weights in weights_by_signal.items():
        for weight in weights:
            if device_type == "SHFQA":
                _save_wave_bin(
                    waves,
                    pulse_map,
                    requires_long_readout,
                    SHFQA_COMPLEX_SAMPLE_SCALING
                    * (weight.samples_i - 1j * weight.samples_q),
                    None,
                    weight.basename,
                    "",
                    device_id=awg_key.device_id,
                    signal_id=signal,
                    downsampling_factor=weight.downsampling_factor,
                )
            else:
                _save_wave_bin(
                    waves,
                    pulse_map,
                    requires_long_readout,
                    weight.samples_i,
                    None,
                    weight.basename,
                    "_i",
                )
                _save_wave_bin(
                    waves,
                    pulse_map,
                    requires_long_readout,
                    weight.samples_q,
                    None,
                    weight.basename,
                    "_q",
                )
    for sampled_waveform in sampled_waveforms:
        sampled_signature = sampled_waveform.signature
        sig_string = sampled_waveform.signature_string
        if device_type != "SHFQA":
            if awg_properties.signal_type == "SINGLE":
                _save_wave_bin(
                    waves,
                    pulse_map,
                    requires_long_readout,
                    sampled_signature.samples_i,
                    sampled_signature.pulse_map,
                    sig_string,
                    "",
                )
                if sampled_signature.samples_marker1 is not None:
                    _save_wave_bin(
                        waves,
                        pulse_map,
                        requires_long_readout,
                        sampled_signature.samples_marker1,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_marker1",
                    )
                if sampled_signature.samples_marker2 is not None:
                    _save_wave_bin(
                        waves,
                        pulse_map,
                        requires_long_readout,
                        sampled_signature.samples_marker2,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_marker2",
                    )
            else:
                _save_wave_bin(
                    waves,
                    pulse_map,
                    requires_long_readout,
                    sampled_signature.samples_i,
                    sampled_signature.pulse_map,
                    sig_string,
                    "_i",
                )
                if sampled_signature.samples_q is not None:
                    _save_wave_bin(
                        waves,
                        pulse_map,
                        requires_long_readout,
                        sampled_signature.samples_q,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_q",
                    )
                if sampled_signature.samples_marker1 is not None:
                    _save_wave_bin(
                        waves,
                        pulse_map,
                        requires_long_readout,
                        sampled_signature.samples_marker1,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_marker1",
                    )
                if sampled_signature.samples_marker2 is not None:
                    _save_wave_bin(
                        waves,
                        pulse_map,
                        requires_long_readout,
                        sampled_signature.samples_marker2,
                        sampled_signature.pulse_map,
                        sig_string,
                        "_marker2",
                    )
        else:
            signal_id = min(sampled_waveform.signals)
            _save_wave_bin(
                waves,
                pulse_map,
                requires_long_readout,
                SHFQA_COMPLEX_SAMPLE_SCALING
                * (sampled_signature.samples_i - 1j * sampled_signature.samples_q),
                sampled_signature.pulse_map,
                sig_string,
                "",
                device_id=awg_key.device_id,
                signal_id=signal_id,
                hold_start=sampled_signature.hold_start,
                hold_length=sampled_signature.hold_length,
            )


def _save_wave_bin(
    # Fields to be saved
    waves: dict[str, CodegenWaveform],
    pulse_map: dict[str, PulseMapEntry],
    requires_long_readout: dict[str, list[str]],
    # Parameters for the wave to be saved
    samples,
    signature_pulse_map,
    sig_string: str,
    suffix: str,
    device_id: str | None = None,
    signal_id: str | None = None,
    hold_start: int | None = None,
    hold_length: int | None = None,
    downsampling_factor: int | None = None,
):
    filename = sig_string + suffix + ".wave"
    wave = CodegenWaveform(
        samples=samples,
        hold_start=hold_start,
        hold_length=hold_length,
        downsampling_factor=downsampling_factor,
    )
    assert filename not in waves or np.allclose(
        waves[filename].samples, wave.samples
    ), filename
    waves[filename] = wave
    if (
        hold_start is not None
        or hold_length is not None
        or downsampling_factor is not None
    ):
        assert device_id is not None
        assert signal_id is not None
        device_long_readout_signals = requires_long_readout.setdefault(device_id, [])
        if signal_id not in device_long_readout_signals:
            device_long_readout_signals.append(signal_id)
    _append_to_pulse_map(pulse_map, signature_pulse_map, sig_string)


def _append_to_pulse_map(pulse_map, signature_pulse_map, sig_string):
    if signature_pulse_map is None:
        return
    for pulse_id, pulse_waveform_map in signature_pulse_map.items():
        pulse_map_entry = pulse_map.setdefault(pulse_id, PulseMapEntry())
        pulse_map_entry.waveforms[sig_string] = pulse_waveform_map
