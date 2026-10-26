# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np

from laboneq.controller.recipe_processor import WaveformItem
from laboneq.controller.utilities.awg_waveform import convert_awg_waveform
from laboneq.core.utilities.replace_phase_increment import calc_ct_replacement
from laboneq.core.utilities.replace_pulse import ReplacementType, calc_wave_replacements

if TYPE_CHECKING:
    from laboneq.controller.devices.device_collection import DeviceCollection
    from laboneq.controller.near_time_replacement import NearTimeReplacements
    from laboneq.controller.recipe_processor import RecipeData
    from laboneq.core.types.enums.wave_type import WaveType
    from laboneq.data.artifacts_qccs import ArtifactsCodegen, CodegenWaveform
    from laboneq.data.nt_step_key import NtStepKey


def process_replacements_qccs(
    artifacts: ArtifactsCodegen,
    recipe_data: RecipeData,
    devices: DeviceCollection,
    nt_step: NtStepKey,
    replacements: NearTimeReplacements,
):
    has_rt_exec_inits = any(
        r.nt_step == nt_step for r in recipe_data.artifacts.realtime_execution_init
    )

    # We can skip replacements right away if new replacements have not been requested
    # and the current NT step completely relies on programs, waveforms, etc. uploaded
    # in previous steps (i.e. has_rt_exec_inits is False for this step). What has been
    # done previously is still in effect.
    if replacements.pulses_updated_after_last_read or has_rt_exec_inits:
        _process_pulse_replacements(
            replacements, recipe_data, artifacts, devices, nt_step
        )
    if replacements.phase_increments_updated_after_last_read or has_rt_exec_inits:
        _process_phase_increment_replacements(
            replacements, recipe_data, devices, nt_step
        )


def _process_pulse_replacements(
    replacements: NearTimeReplacements,
    recipe_data: RecipeData,
    artifacts: ArtifactsCodegen,
    devices: DeviceCollection,
    nt_step: NtStepKey,
):
    nt_step_program_refs = recipe_data.get_program_refs()[nt_step.indices]
    current_waves: dict[str, CodegenWaveform] = {}
    with replacements.pulses() as pulses:
        for pulse_uid, replacement in pulses.items():
            wave_replacements = calc_wave_replacements(
                artifacts,
                pulse_uid,
                replacement,
                current_waves,
            )

            # Ensured by `calc_wave_replacements`
            assert artifacts.wave_indices is not None

            acquisition_type = recipe_data.rt_execution_info.acquisition_type
            for repl in wave_replacements:
                awg_indices = next(
                    cast("dict[str, tuple[int, WaveType]]", a["value"])
                    for a in artifacts.wave_indices
                    if a["filename"] == repl.awg_id
                )
                target_wave_index = awg_indices[repl.sig_string][0]
                seqc_name = repl.awg_id
                if seqc_name not in nt_step_program_refs:
                    # Skip replacement if it is related to a different NT step
                    continue
                awg_key = recipe_data.awg_by_seqc_name(seqc_name)
                assert awg_key is not None
                device = devices.find_by_uid(awg_key.device_uid)

                if repl.replacement_type == ReplacementType.I_Q:
                    assert isinstance(repl.samples, list)
                    clipped = np.clip(repl.samples, -1.0, 1.0)
                    bin_wave = convert_awg_waveform()(*clipped)
                    device.add_waveform_replacement(
                        awg_index=awg_key.awg_index,
                        wave=WaveformItem(
                            index=target_wave_index,
                            name=repl.sig_string + " (repl)",
                            samples=bin_wave,
                        ),
                        acquisition_type=acquisition_type,
                    )
                elif repl.replacement_type == ReplacementType.COMPLEX:
                    assert isinstance(repl.samples, np.ndarray)
                    np.clip(repl.samples.real, -1.0, 1.0, out=repl.samples.real)
                    np.clip(repl.samples.imag, -1.0, 1.0, out=repl.samples.imag)
                    device.add_waveform_replacement(
                        awg_index=awg_key.awg_index,
                        wave=WaveformItem(
                            index=target_wave_index,
                            name=repl.sig_string + " (repl)",
                            samples=repl.samples,
                        ),
                        acquisition_type=acquisition_type,
                    )


def _process_phase_increment_replacements(
    replacements: NearTimeReplacements,
    recipe_data: RecipeData,
    devices: DeviceCollection,
    nt_step: NtStepKey,
):
    nt_step_program_refs = recipe_data.get_program_refs()[nt_step.indices]

    with replacements.phase_increments() as phase_increments:
        for parameter_uid, replacement in phase_increments.items():
            ct_replacements = calc_ct_replacement(
                recipe_data.artifacts, parameter_uid, replacement
            )
            for repl in ct_replacements:
                seqc_name = repl["seqc"]
                if seqc_name not in nt_step_program_refs:
                    # Skip replacement if it is related to a different NT step
                    continue
                awg_key = recipe_data.awg_by_seqc_name(seqc_name)
                assert awg_key is not None
                device = devices.find_by_uid(awg_key.device_uid)
                device.add_command_table_replacement(awg_key.awg_index, repl["ct"])
