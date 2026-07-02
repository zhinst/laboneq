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
    ResultSource,
    WeightInfo,
)

if TYPE_CHECKING:
    from laboneq._rust import codegenerator as codegen_rs


def generate_output(output: codegen_rs.SeqCGenOutput) -> SeqCGenOutput:
    integration_unit_allocations = {}
    awg_properties = {}
    channel_properties = defaultdict(list)
    integration_times = {}
    result_lengths = {}
    integration_weights = {}
    src = {}
    feedback_register_config = {}
    wave_indices_all = {}
    command_tables = {}
    parameter_phase_increment_map = {}

    waves = dict(output.waves)
    requires_long_readout = defaultdict(list, output.requires_long_readout)
    pulse_map = output.pulse_map

    for awg_output in output.awg_results:
        awg_key = AwgKey(*awg_output.awg_properties.key)
        integration_unit_allocations[awg_key] = awg_output.integration_unit_allocations

        awg_properties[awg_key] = awg_output.awg_properties
        channel_properties[awg_key].extend(awg_output.channel_properties)
        for signal, integration_time in awg_output.integration_lengths.items():
            integration_times[signal] = SignalIntegrationInfo(
                is_play=integration_time.is_play,
                length_in_samples=integration_time.length,
            )
        if awg_output.result_length is not None:
            result_lengths[awg_key] = awg_output.result_length
        integration_weights[awg_key] = [
            WeightInfo(
                id=iw.basename,
                integration_units=iw.integration_units,
                downsampling_factor=iw.downsampling_factor,
            )
            for iw in awg_output.integration_weights
        ]

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
