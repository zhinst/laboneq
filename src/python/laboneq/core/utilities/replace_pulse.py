# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.core.types.enums.wave_type import WaveType
from laboneq.core.utilities.pulse_sampler import (
    combine_pulse_parameters,
    sample_pulse,
    verify_amplitude_no_clipping,
)
from laboneq.data.scheduled_experiment import (
    ArtifactsCodegen,
    CodegenWaveform,
    CompilerArtifact,
    PulseWaveformMap,
)

if TYPE_CHECKING:
    from laboneq.core.types.compiled_experiment import CompiledExperiment
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.session import Session

_logger = logging.getLogger(__name__)


class Component(Enum):
    REAL = auto()
    IMAG = auto()
    COMPLEX = auto()


def _replace_pulse_in_wave(
    artifacts: ArtifactsCodegen,
    wave_name: str,
    pulse_or_array: ArrayLike | Pulse,
    pwm: PulseWaveformMap,
    current_waves: dict[str, CodegenWaveform] | None,
    component: Component = Component.COMPLEX,
    is_complex: bool = True,
):
    current_wave = None
    if current_waves is not None:
        current_wave = current_waves.get(wave_name)
    if current_wave is None:
        specified_wave = artifacts.waves[wave_name]
        # TODO(2K): Avoid deepcopy on every iteration, create working copy once per execution
        current_wave = deepcopy(specified_wave)
        if current_waves is not None:
            current_waves[wave_name] = current_wave

    new_samples = current_wave.samples  # Reference
    input_samples = None
    amplitude = 1.0
    function = None
    pulse_id = "<replaced pulse>"
    pulse_parameters: dict[str, Any] = {}
    if isinstance(pulse_or_array, list):
        pulse_or_array = np.asarray(pulse_or_array)
    if isinstance(pulse_or_array, np.ndarray):
        input_samples = pulse_or_array
    else:
        input_samples = getattr(pulse_or_array, "samples", None)
        amplitude = getattr(pulse_or_array, "amplitude", 1.0)
        function = getattr(pulse_or_array, "function", None)
        pulse_id = getattr(pulse_or_array, "uid", pulse_id)
        pulse_parameters = getattr(pulse_or_array, "pulse_parameters", {})

    if (
        not is_complex
        and input_samples is not None
        and np.iscomplex(input_samples).any()
    ):
        raise LabOneQException("Complex numbers found in non-complex pulse replacement")

    for instance in pwm.instances:
        if not is_complex and (
            (instance.channel == 0) != (component == Component.REAL)
            and instance.channel is not None
        ):
            continue
        phase = instance.iq_phase or 0.0
        combined_pulse_parameters = combine_pulse_parameters(
            initial_pulse=instance.pulse_pulse_parameters,
            replaced_pulse=pulse_parameters,
            play=instance.play_pulse_parameters,
        )
        samples = sample_pulse(
            signal_type=pwm.signal_type,
            sampling_rate=pwm.sampling_rate,
            length=(
                instance.length if instance.length is not None else pwm.length_samples
            )
            / pwm.sampling_rate,
            amplitude=amplitude * instance.amplitude,
            pulse_function=function,
            modulation_frequency=instance.modulation_frequency,
            phase=phase,
            samples=input_samples,
            mixer_type=pwm.mixer_type,
            pulse_parameters=combined_pulse_parameters,
        )
        verify_amplitude_no_clipping(
            samples_i=samples["samples_i"],
            samples_q=samples["samples_q"],
            pulse_id=pulse_id,
            mixer_type=pwm.mixer_type,
            signals=None,
            is_rf_signal=pwm.signal_type != "iq",
        )

        if "samples_q" in samples and instance.needs_conjugate:
            samples["samples_q"] = -samples["samples_q"]

        len_samples = len(samples["samples_i"])
        if instance.offset_samples < 0:
            pulse_ofs = -instance.offset_samples
            target_ofs = 0
            plen = len_samples - pulse_ofs
        else:
            pulse_ofs = 0
            target_ofs = instance.offset_samples
            plen = min(len_samples, len(new_samples) - target_ofs)
        if component == Component.COMPLEX:
            new_samples[target_ofs : target_ofs + plen] = (
                samples["samples_i"][pulse_ofs : pulse_ofs + plen]
                - 1.0j * samples["samples_q"][pulse_ofs : pulse_ofs + plen]
            )
        else:
            comp = (
                "samples_i"
                if component == Component.REAL or not is_complex
                else "samples_q"
            )
            new_samples[target_ofs : target_ofs + plen] = samples[comp][
                pulse_ofs : pulse_ofs + plen
            ]
    return new_samples


class ReplacementType(Enum):
    I_Q = auto()
    COMPLEX = auto()


@dataclass
class WaveReplacement:
    awg_id: str
    sig_string: str
    replacement_type: ReplacementType
    samples: np.ndarray | list[np.ndarray]


def calc_wave_replacements(
    artifacts: CompilerArtifact | None,
    pulse_uid: str | Pulse,
    pulse_or_array: ArrayLike | Pulse,
    current_waves: dict[str, CodegenWaveform] | None = None,
) -> list[WaveReplacement]:
    if not isinstance(artifacts, ArtifactsCodegen):
        raise LabOneQException("Wave replacement is not supported for this instrument.")

    if artifacts.pulse_map is None or artifacts.wave_indices is None:
        raise LabOneQException("No waves available for replacement.")

    eff_pulse_uid = (
        pulse_uid
        if isinstance(pulse_uid, str)
        else getattr(pulse_uid, "uid", "<unknown>")
    )

    pm = artifacts.pulse_map.get(eff_pulse_uid)
    if pm is None:
        _logger.warning("No mapping found for pulse '%s' - ignoring", eff_pulse_uid)
        return []

    for waveform in pm.waveforms.values():
        if any([instance.can_compress for instance in waveform.instances]):
            raise LabOneQException(
                f"Pulse replacement is not allowed for pulses that support compression. Pulse '{eff_pulse_uid}'."
            )

    replacements: list[WaveReplacement] = []
    for sig_string, pwm in pm.waveforms.items():
        for awg_wave_map in artifacts.wave_indices:
            target_wave = cast(
                tuple[int, WaveType] | None, awg_wave_map["value"].get(sig_string)
            )
            if target_wave is None or len(target_wave) != 2:
                continue
            _, sig_type = target_wave  # unpack
            if sig_type == WaveType.SINGLE:
                replacement_type = ReplacementType.I_Q
                samples_i = _replace_pulse_in_wave(
                    artifacts,
                    sig_string + ".wave",
                    pulse_or_array,
                    pwm,
                    current_waves=current_waves,
                    component=Component.REAL,
                    is_complex=False,
                )
                samples = [samples_i]
            elif sig_type != WaveType.COMPLEX:
                is_complex = sig_type == WaveType.IQ
                replacement_type = ReplacementType.I_Q
                samples_i = _replace_pulse_in_wave(
                    artifacts,
                    sig_string + "_i.wave",
                    pulse_or_array,
                    pwm,
                    current_waves=current_waves,
                    component=Component.REAL,
                    is_complex=is_complex,
                )
                samples_q = _replace_pulse_in_wave(
                    artifacts,
                    sig_string + "_q.wave",
                    pulse_or_array,
                    pwm,
                    current_waves=current_waves,
                    component=Component.IMAG,
                    is_complex=is_complex,
                )
                samples = [samples_i, samples_q]
            else:
                replacement_type = ReplacementType.COMPLEX
                samples = _replace_pulse_in_wave(
                    artifacts,
                    sig_string + ".wave",
                    pulse_or_array,
                    pwm,
                    current_waves=current_waves,
                    component=Component.COMPLEX,
                    is_complex=True,
                )
            replacements.append(
                WaveReplacement(
                    awg_id=cast(str, awg_wave_map["filename"]),
                    sig_string=sig_string,
                    replacement_type=replacement_type,
                    samples=samples,
                )
            )

    return replacements


def replace_pulse(
    target: CompiledExperiment | Session,
    pulse_uid: str | Pulse,
    pulse_or_array: ArrayLike | Pulse,
):
    """Replaces specific pulse with the new sample data.

    Args:
        target: CompiledExperiment or Session.
                See CompiledExperiment.replace_pulse and Session.replace_pulse for details.
        pulse_uid: pulse to replace, can be Pulse object or uid of the pulse
        pulse_or_array: replacement pulse, can be Pulse object or value array (see sampled_pulse_* from the pulse library)
    """
    from laboneq.core.types.compiled_experiment import CompiledExperiment

    if isinstance(target, CompiledExperiment):
        artifacts = target.scheduled_experiment.artifacts
        wave_replacements = calc_wave_replacements(artifacts, pulse_uid, pulse_or_array)
        # Ensured by `calc_wave_replacements`
        assert isinstance(artifacts, ArtifactsCodegen)
        for repl in wave_replacements:
            if repl.replacement_type == ReplacementType.I_Q:
                if len(repl.samples) == 2:
                    wave_i = artifacts.waves[repl.sig_string + "_i.wave"]
                    wave_q = artifacts.waves[repl.sig_string + "_q.wave"]
                    wave_i.samples = repl.samples[0]
                    wave_q.samples = repl.samples[1]
                else:
                    artifacts.waves[repl.sig_string + ".wave"].samples = repl.samples[0]
    else:
        target.replace_pulse(pulse_uid, pulse_or_array)
