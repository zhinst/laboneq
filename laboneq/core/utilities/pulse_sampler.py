# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Union, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import UserFunction

_logger = logging.getLogger(__name__)


def length_to_samples(length, sampling_rate):
    return round(length * sampling_rate)


def interval_to_samples(start, end, sampling_rate):
    start_samples = length_to_samples(start, sampling_rate)
    end_samples = start_samples + length_to_samples(end - start, sampling_rate)
    return start_samples, end_samples


def sample_pulse(
    *,
    signal_type: str,
    sampling_rate: float,
    length: float,
    amplitude: Union[float, complex],
    pulse_function: Optional[str],
    user_function: Optional[UserFunction] = None,
    modulation_frequency: Optional[float] = None,
    modulation_phase: Optional[float] = None,
    iq_phase: float = 0,
    samples: Optional[np.ndarray] = None,
    complex_modulation: bool = True,
    pulse_parameters: Optional[Dict[str, Any]] = None,
):
    """Create a waveform from a pulse definition.

    Depending on the given parameters, this function either samples a functional
    pulse or applies amplitude and phase parameters to an array of samples.

    Only one of samples and pulse_function may be given at the
    same time. Software modulation is performed when modulation_frequency and
    modulation_phase are not None.

    Args:
        signal_type: "iq" if the pulse represents quadrature (IQ) moduluation; used
          together with samples.
        sampling_rate: Sampling rate of the device the pulse is played on.
        length: Pulse length in seconds
        amplitude: Magnitude of the amplitude to multiply with the given pulse
        iq_phase: Phase of the amplitude to multiply with the given pulse
        pulse_function: In case of a functional pulse, the function to sample
        user_function: User provided function to generate samples
        modulation_frequency: The oscillator frequency (for software modulation if
          not None)
        modulation_phase: The oscillator phase (for software modulation if not
          None)
        samples: Pulse shape for a sampled pulse
        complex_modulation: Whether to allow applying iq_phase to the complex
          samples (False to emulate UHFQA behavior)
        pulse_parameters: resolved parameters passed to the user pulse function

    Returns:
        A dict with one ("samples_i", real case) or two ("samples_i" and
        "samples_q", complex case) ndarrays representing the sampled, scaled and
        possibly modulated waveform for this pulse.

    Raises:
        ValueError: Invalid combination of arguments
    """
    if pulse_function == "":
        pulse_function = None

    if samples is not None and pulse_function is not None and user_function is not None:
        raise ValueError(
            "Only one of samples, pulse_function or user_function may be given at the same time."
        )

    num_samples = length_to_samples(length, sampling_rate)

    if pulse_function is not None:
        samples = pulse_function_library[pulse_function](
            length=length, num_samples=num_samples, sampling_rate=sampling_rate
        )
    if user_function is not None:
        samples = user_function(
            np.linspace(-1, 1, num_samples),
            pulse_parameters={
                "length": length,
                "amplitude": amplitude,
                "sampling_rate": sampling_rate,
                **dict(
                    sorted(
                        ({} if pulse_parameters is None else pulse_parameters).items()
                    )
                ),
            },
        )
    samples = np.array(samples[:num_samples])
    shape = samples.shape
    if len(shape) > 1:
        assert len(shape) == 2 and shape[1] == 2
        samples = samples[:, 0] + 1j * samples[:, 1]

    if signal_type == "iq":
        samples = samples.astype(complex)
        if (complex_modulation or pulse_function) and iq_phase:
            amplitude *= np.exp(1j * iq_phase)
    else:
        assert all(samples.imag == 0.0)
        amplitude = amplitude.real
        samples = samples.real

    samples *= amplitude

    if modulation_frequency is not None:
        _logger.debug(
            "Doing modulation with modulation_frequency %s and phase %s",
            modulation_frequency,
            modulation_phase,
        )

        phase = (
            2.0 * np.pi * modulation_frequency * np.arange(num_samples) / sampling_rate
        ) + (modulation_phase or 0.0)

        if signal_type == "iq":
            if complex_modulation:
                samples = np.exp(-1.0j * phase) * samples
            else:
                samples = (
                    np.cos(phase) * samples.real - 1j * np.sin(phase) * samples.imag
                )
        else:
            samples = np.cos(phase) * samples

    return {"samples_i": samples.real, "samples_q": samples.imag}


def const(length, num_samples, sampling_rate):
    return np.ones(num_samples)


def gaussian(length, num_samples, sampling_rate):
    sigma = calc_sigma(length, sampling_rate)
    shift = float(length * sampling_rate) / 2
    ts = np.arange(num_samples)
    return gauss(ts, sigma, shift)


pulse_function_library = dict()
pulse_function_library["const"] = const
pulse_function_library["gaussian"] = gaussian


def gauss(t, sigma, shift):
    shift_t = t - shift
    return np.exp(-shift_t * shift_t / (2 * sigma * sigma))


def calc_sigma(length, sampling_rate):
    return length * sampling_rate / 6
