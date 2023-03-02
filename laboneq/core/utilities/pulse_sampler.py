# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.mixer_type import MixerType

_logger = logging.getLogger(__name__)


def length_to_samples(length, sampling_rate):
    return round(length * sampling_rate)


def interval_to_samples(start, end, sampling_rate):
    start_samples = length_to_samples(start, sampling_rate)
    end_samples = start_samples + length_to_samples(end - start, sampling_rate)
    return start_samples, end_samples


def interval_to_samples_with_errors(start, end, sampling_rate):
    start_samples = length_to_samples(start, sampling_rate)
    end_samples = start_samples + length_to_samples(end - start, sampling_rate)
    start_rounding_error = start - start_samples / sampling_rate
    end_rounding_error = end - end_samples / sampling_rate

    return (start_samples, end_samples), (start_rounding_error, end_rounding_error)


def sample_pulse(
    *,
    signal_type: str,
    sampling_rate: float,
    length: float,
    amplitude: Union[float, complex],
    pulse_function: Optional[str],
    modulation_frequency: Optional[float] = None,
    phase: Optional[float] = None,
    samples: Optional[np.ndarray] = None,
    mixer_type: Optional[MixerType] = MixerType.IQ,
    pulse_parameters: Optional[Dict[str, Any]] = None,
    markers=None,
):
    """Create a waveform from a pulse definition.

    Depending on the given parameters, this function either samples a functional
    pulse or applies amplitude and phase parameters to an array of samples.

    Only one of samples and pulse_function may be given at the
    same time. Software modulation is performed when modulation_frequency and
    modulation_phase are not None.

    Depending on the following mixer stage, the output of the AWG should either be real
    and imaginary components of a complex signal (IQ modulation, default), or
    UHFQA-style envelope modulation, where both AWG channels should output the signal's
    envelope. Specify the type via the `mixer_type` argument.

    Args:
        signal_type: "iq" if the pulse represents quadrature (IQ) modulation; used
          together with samples.
        sampling_rate: Sampling rate of the device the pulse is played on.
        length: Pulse length in seconds
        amplitude: Magnitude of the amplitude to multiply with the given pulse
        pulse_function: In case of a functional pulse, the function to sample
        modulation_frequency: The oscillator frequency (for software modulation if
          not None)
        phase: The phase shift to apply to the signal
        samples: Pulse shape for a sampled pulse
        mixer_type: Type of the mixer after the AWG. Only effective for IQ signals.

    Returns:
        A dict with one ("samples_i", real case) or two ("samples_i" and
        "samples_q", complex case) ndarrays representing the sampled, scaled and
        possibly modulated waveform for this pulse.

    Raises:
        ValueError: Invalid combination of arguments
    """
    if pulse_function == "":
        pulse_function = None

    if samples is not None and pulse_function is not None:
        raise ValueError(
            "Only one of samples or pulse_function may be given at the same time."
        )

    num_samples = length_to_samples(length, sampling_rate)

    if pulse_function is not None:
        samples = pulse_function_library[pulse_function](
            np.linspace(-1, 1, num_samples, endpoint=False),
            length=length,
            amplitude=amplitude,
            sampling_rate=sampling_rate,
            **(pulse_parameters or {}),
        )
    samples = np.array(samples[:num_samples])
    shape = samples.shape
    if len(shape) > 1:
        assert len(shape) == 2 and shape[1] == 2
        samples = samples[:, 0] + 1j * samples[:, 1]

    if signal_type == "iq":
        samples = samples.astype(complex)
    else:
        assert all(samples.imag == 0.0)
        amplitude = amplitude.real
        samples = samples.real

    samples *= amplitude

    phase = phase or 0.0
    modulation_frequency = modulation_frequency or 0.0

    _logger.debug(
        "Doing modulation with modulation_frequency %f and phase %f",
        modulation_frequency,
        phase,
    )

    t = np.arange(num_samples) / sampling_rate
    if modulation_frequency:
        carrier_phase = 2.0 * np.pi * modulation_frequency * t
    else:
        carrier_phase = 0
    carrier_phase += phase

    if signal_type == "iq":
        samples = np.exp(-1.0j * carrier_phase) * samples
    else:
        if not np.allclose(samples.imag, 0.0):
            raise LabOneQException("Complex samples not permitted for RF signals")
        samples = np.cos(carrier_phase) * samples

    if mixer_type == MixerType.UHFQA_ENVELOPE and signal_type == "iq":
        if not np.allclose(samples.imag, 0.0):
            raise LabOneQException(
                "HW modulation on UHFQA requires a real baseband (phase "
                "modulation is not permitted)."
            )
        samples = samples.real * (1.0 + 1.0j)

    retval = {"samples_i": samples.real, "samples_q": samples.imag}
    if markers:
        if next(
            (
                True
                for m in markers
                if m.get("marker_selector") == "marker1" and m.get("enable")
            ),
            False,
        ):  #  and m.get("enabled"), False):
            samples_markers = np.ones_like(samples, dtype=np.int16)
            retval["samples_marker1"] = samples_markers
        if next(
            (
                m.get("enable")
                for m in markers
                if m.get("marker_selector") == "marker2" and m.get("enable")
            ),
            False,
        ):

            samples_markers = np.ones(len(samples), dtype=np.int16)
            retval["samples_marker2"] = samples_markers

    return retval


pulse_function_library = dict()


def verify_amplitude_no_clipping(
    samples, pulse_id: Optional[str], mixer_type: MixerType, signal_id: Optional[str]
):
    max_amplitude = np.max(
        np.abs(samples["samples_i"] + 1j * samples.get("samples_j", 0))
    )
    if mixer_type == MixerType.UHFQA_ENVELOPE:
        max_amplitude /= np.sqrt(2)
    TOLERANCE = 1e-6
    if max_amplitude > 1 + TOLERANCE:
        if pulse_id is not None:
            message = (
                f"Pulse '{pulse_id}' {f'on signal {signal_id} ' if signal_id else ''}"
                f"exceeds the max allowed amplitude."
            )
        else:
            message = (
                f"A waveform on signal '{signal_id}' exceeds the max allowed amplitude."
            )

        message += " Signal will be clipped on the device."
        _logger.warning(message)


def combine_pulse_parameters(initial_pulse, replaced_pulse, play):
    combined_parameters = deepcopy(initial_pulse) or {}
    if replaced_pulse is not None:
        combined_parameters.update(replaced_pulse)
    if play is not None:
        combined_parameters.update(play)
    return combined_parameters
