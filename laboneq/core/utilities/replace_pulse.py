# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Union, Optional, TYPE_CHECKING
import logging
import numpy as np
from numpy.typing import ArrayLike
from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.core.utilities.pulse_sampler import sample_pulse

if TYPE_CHECKING:
    from laboneq.core.types.compiled_experiment import (
        CompiledExperiment,
        PulseWaveformMap,
    )
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.session import Session

_logger = logging.getLogger(__name__)


class Component(Enum):
    REAL = auto()
    IMAG = auto()
    COMPLEX = auto()


def _replace_pulse_in_wave(
    compiled_experiment: CompiledExperiment,
    wave_name: str,
    pulse_or_array: Union[ArrayLike, Pulse],
    pwm: PulseWaveformMap,
    component: Component = Component.COMPLEX,
    is_complex: bool = True,
    current_waves: Optional[List] = None,
):
    current_wave = None
    if current_waves is not None:
        current_wave = next(
            (w for w in current_waves if w["filename"] == wave_name), None,
        )
    if current_wave is None:
        specified_wave = next(
            w for w in compiled_experiment.waves if w["filename"] == wave_name
        )
        # TODO(2K): Avoid deepcopy on every iteration, create working copy once per execution
        current_wave = deepcopy(specified_wave)
        if current_waves is not None:
            current_waves.append(current_wave)

    new_samples = current_wave["samples"]  # Reference
    input_samples = None
    amplitude = 1.0
    function = None
    length = None
    if isinstance(pulse_or_array, list):
        pulse_or_array = np.array(pulse_or_array)
    if isinstance(pulse_or_array, np.ndarray):
        input_samples = pulse_or_array
    else:
        length = int(getattr(pulse_or_array, "length", 0) * pwm.sampling_rate)
        input_samples = getattr(pulse_or_array, "samples", None)
        amplitude = getattr(pulse_or_array, "amplitude", 1.0)
        function = getattr(pulse_or_array, "function", None)
        if function is not None:
            function = function.value
    length = length or (0 if input_samples is None else len(input_samples))
    assert length > 0 and abs(length - pwm.length_samples) < 2  # (rounding)

    if (
        not is_complex
        and input_samples is not None
        and np.iscomplex(input_samples).any()
    ):
        raise LabOneQException("Complex numbers found in non-complex pulse replacement")

    for instance in pwm.instances:
        if not is_complex and (instance.channel == 0) != (component == Component.REAL):
            continue
        samples = sample_pulse(
            signal_type=pwm.signal_type,
            sampling_rate=pwm.sampling_rate,
            length=pwm.length_samples / pwm.sampling_rate,
            amplitude=amplitude * instance.amplitude,
            pulse_function=function,
            modulation_frequency=instance.modulation_frequency,
            modulation_phase=instance.modulation_phase,
            iq_phase=instance.iq_phase,
            samples=input_samples,
            complex_modulation=pwm.complex_modulation,
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
    samples: List[ArrayLike]


def calc_wave_replacements(
    compiled_experiment: CompiledExperiment,
    pulse_uid: Union[str, Pulse],
    pulse_or_array: Union[ArrayLike, Pulse],
    current_waves: Optional[List] = None,
) -> List[WaveReplacement]:
    if not isinstance(pulse_uid, str):
        pulse_uid = pulse_uid.uid
    pm = compiled_experiment.pulse_map.get(pulse_uid)
    if pm is None:
        _logger.warning("No mapping found for pulse %s - ignoring", pulse_uid)
        return

    replacements: List[WaveReplacement] = []
    for sig_string, pwm in pm.waveforms.items():
        for awgs in compiled_experiment.wave_indices:
            awg_wave_map = awgs["value"]
            target_wave = awg_wave_map.get(sig_string)
            if target_wave is None:
                continue
            wave_type = target_wave[1]
            if wave_type not in ("iq", "double", "multi", "complex"):
                raise LabOneQException(
                    f"Pulse replacement for the waves of type '{wave_type}' is not yet supported"
                )
            is_complex = wave_type == "iq" or wave_type == "complex"
            if wave_type != "complex":
                replacement_type = ReplacementType.I_Q
                samples_i = _replace_pulse_in_wave(
                    compiled_experiment,
                    sig_string + "_i.wave",
                    pulse_or_array,
                    pwm,
                    component=Component.REAL,
                    is_complex=is_complex,
                    current_waves=current_waves,
                )
                samples_q = _replace_pulse_in_wave(
                    compiled_experiment,
                    sig_string + "_q.wave",
                    pulse_or_array,
                    pwm,
                    component=Component.IMAG,
                    is_complex=is_complex,
                    current_waves=current_waves,
                )
                samples = [samples_i, samples_q]
            else:
                replacement_type = ReplacementType.COMPLEX
                samples = _replace_pulse_in_wave(
                    compiled_experiment,
                    sig_string + ".wave",
                    pulse_or_array,
                    pwm,
                    component=Component.COMPLEX,
                    is_complex=is_complex,
                    current_waves=current_waves,
                )
            replacements.append(
                WaveReplacement(
                    awgs["filename"], sig_string, replacement_type, samples,
                )
            )

    return replacements


def replace_pulse(
    target: Union[CompiledExperiment, Session],
    pulse_uid: Union[str, Pulse],
    pulse_or_array: Union[ArrayLike, Pulse],
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
        wave_replacements = calc_wave_replacements(target, pulse_uid, pulse_or_array)
        for repl in wave_replacements:
            if repl.replacement_type == ReplacementType.I_Q:
                wave_i = next(
                    w
                    for w in target.waves
                    if w["filename"] == repl.sig_string + "_i.wave"
                )
                wave_q = next(
                    w
                    for w in target.waves
                    if w["filename"] == repl.sig_string + "_q.wave"
                )
                wave_i["samples"] = repl.samples[0]
                wave_q["samples"] = repl.samples[1]
    else:
        target.replace_pulse(pulse_uid, pulse_or_array)
