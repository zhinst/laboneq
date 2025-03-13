# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from copy import deepcopy
from numbers import Complex
from typing import Any, Callable, cast

import numpy as np

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums.mixer_type import MixerType
from laboneq.data.compilation_job import PulseDef

_logger = logging.getLogger(__name__)


def length_to_samples(length, sampling_rate) -> int:
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


def sample_marker(
    num_total_samples, sampling_rate, enable, start, length, pulse_def: PulseDef
):
    """Sample a marker.

    Args:
        num_total_samples: Number of samples in the pulse
        sampling_rate: The sampling rate
        enable: Whether the marker is fully enabled
        start: Start time of the marker
        length: Length of the marker
        pulse_def: The pulse definition used for sampling the marker waveform

    Returns:
        A numpy array of the marker samples
    """
    if enable:
        return np.ones(num_total_samples, dtype=np.uint8)
    if pulse_def is not None:
        if pulse_def.samples is not None:
            num_samples_marker = len(pulse_def.samples)
            if num_samples_marker != num_total_samples:
                raise LabOneQException(
                    f"The pulse and marker waveforms must have the same length, "
                    f"but currently the pulse has {num_total_samples} samples while "
                    f"the marker has {num_samples_marker} samples."
                )
            # in this case `length` is None, so derive it from the number of samples
            length = num_samples_marker / sampling_rate
        marker_samples = sample_pulse(
            signal_type="marker",
            sampling_rate=sampling_rate,
            length=length,
            amplitude=1.0,
            pulse_function=pulse_def.function,
            samples=pulse_def.samples,
            mixer_type=None,
            markers=None,
        )
        marker_samples_int = np.clip(marker_samples["samples_i"], 0, 1).astype(np.uint8)

        return marker_samples_int

    if start is None:
        return None

    if length is None:
        length = num_total_samples / sampling_rate - start

    start_samples = length_to_samples(start, sampling_rate)
    end_samples = start_samples + length_to_samples(length, sampling_rate)
    marker_samples = np.zeros(num_total_samples, dtype=np.uint8)
    marker_samples[start_samples:end_samples] = 1
    return marker_samples


def sample_pulse(
    *,
    signal_type: str,
    sampling_rate: float,
    length: float,
    amplitude: Complex,
    pulse_function: str | None,
    modulation_frequency: float | None = None,
    phase: float | None = None,
    samples: np.ndarray | None = None,
    mixer_type: MixerType | None = MixerType.IQ,
    pulse_parameters: dict[str, Any] | None = None,
    markers: list[dict[str, Any]] | None = None,
    pulse_defs: dict[str, PulseDef] | None = None,
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
        pulse_parameters: Extra pulse parameters, passed to the sampler,
        markers: Configuration of the markers,
        pulse_defs: Definitions of other pulses (may be referenced by marker spec),
    Returns:
        A dict with one ("samples_i", real case) or two ("samples_i" and
        "samples_q", complex case) ndarrays representing the sampled, scaled and
        possibly modulated waveform for this pulse. Markers are optionally represented
        by more fields ("marker1" and maybe "marker2")

    Raises:
        ValueError: Invalid combination of arguments
    """
    markers = cast(
        list, {} if markers is None else markers
    )  # Better to never be None, but here we are.
    if pulse_function == "":
        pulse_function = None

    if samples is not None and pulse_function is not None:
        raise ValueError(
            "Only one of samples or pulse_function may be given at the same time."
        )

    num_samples = length_to_samples(length, sampling_rate)
    pulse_parameters = pulse_parameters or {}
    if "amplitude" in pulse_parameters:
        amplitude *= pulse_parameters["amplitude"]
        pulse_parameters = {
            k: v for (k, v) in pulse_parameters.items() if k != "amplitude"
        }

    if pulse_function is not None:
        samples = pulse_function_library[pulse_function](
            np.linspace(-1, 1, num_samples, endpoint=False),
            length=length,
            amplitude=amplitude,
            sampling_rate=sampling_rate,
            **pulse_parameters,
        )
    elif samples is None:
        raise AssertionError("Must provide either samples or a function")

    assert isinstance(samples, (list, np.ndarray))
    samples = np.array(samples[:num_samples])
    shape = samples.shape
    if len(shape) > 1:
        assert len(shape) == 2 and shape[1] == 2
        samples = samples[:, 0] + 1j * samples[:, 1]

    if signal_type == "iq":
        samples = samples.astype(complex)
    else:
        if not all(samples.imag == 0.0):
            _logger.info(
                "Complex-valued pulse envelope provided for an rf-signal, imaginary part will be dropped."
            )
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
        samples = np.cos(carrier_phase) * samples

    if mixer_type == MixerType.UHFQA_ENVELOPE and signal_type == "iq":
        if not np.allclose(samples.imag, 0.0):
            raise LabOneQException(
                "HW modulation on UHFQA requires a real baseband (phase "
                "modulation is not permitted)."
            )
        samples = samples.real * (1.0 + 1.0j)

    retval = {"samples_i": samples.real, "samples_q": samples.imag}

    for m in markers:
        marker_pulse_id = m.get("pulse_id")
        if marker_pulse_id is not None:
            marker_pulse_def = pulse_defs[marker_pulse_id]
            if m.get("length") is not None:
                raise LabOneQException(
                    "Specifying both waveform and length for the markers is not "
                    "supported. Please set markers either via "
                    "{'marker': {'waveform': marker_pulse}} or "
                    "{'marker': {'start': 0, 'length': marker_length}}."
                )
            marker_length = marker_pulse_def.length
            if marker_length is not None and marker_length != length:
                raise LabOneQException(
                    f"The pulse and marker waveforms must have the same length, "
                    f"but currently the pulse has length {length} while the "
                    f"marker has length {marker_length}."
                )
        else:
            marker_pulse_def = None
            if (marker_length := m.get("length")) is None:
                marker_length = length
        m_sampled = sample_marker(
            len(samples),
            sampling_rate=sampling_rate,
            enable=m.get("enable"),
            start=m.get("start"),
            length=marker_length,
            pulse_def=marker_pulse_def,
        )
        if m_sampled is not None:
            marker_selector = m["marker_selector"]
            retval[f"samples_{marker_selector}"] = m_sampled

    return retval


# registries of pulse samplers and factories:
_pulse_samplers: dict[str, Callable[..., Any]] = {}
_pulse_factories: dict[str, Callable[..., Any]] = {}

# deprecated alias for _pulse_samples, use pulse_library.pulse_sampler(...) instead:
pulse_function_library = _pulse_samplers


def verify_amplitude_no_clipping(
    samples, pulse_id: str | None, mixer_type: MixerType | None, signal_id: str | None
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
