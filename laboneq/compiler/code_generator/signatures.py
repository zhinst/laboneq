# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

import numpy as np
from orjson import orjson

from laboneq.compiler.code_generator.utils import normalize_phase
from laboneq.core.utilities.string_sanitize import string_sanitize


@dataclass(unsafe_hash=True)
class PulseSignature:
    """Signature of a single pulse, part of a sampled waveform"""

    start: int  #: the offset of the pulse in the waveform
    pulse: str  #: the pulse function
    length: int  #: the length of the pulse in samples
    amplitude: Optional[float]  #: the amplitude of the pulse
    phase: Optional[float]  #: the phase of the pulse
    #: the oscillator phase of the pulse (for SW oscillators)
    oscillator_phase: Optional[float]
    #: the oscillator frequency of the pulse (for SW oscillators)
    oscillator_frequency: Optional[float]
    baseband_phase: Optional[float]  #: phase offsets from `set_oscillator_phase`
    channel: Optional[int]  #: the channel of the pulse (for HDAWG)
    sub_channel: Optional[int]  #: the sub-channel of the pulse (for SHFQA)
    pulse_parameters: FrozenSet[Tuple[str, str]]  #: additional user pulse parameters
    markers: Any  #: markers played during this pulse


@dataclass(frozen=True)
class SamplesSignature:
    """A collection of samples. Defines a WaveformSignature after compression. See also docstring of WaveformSignature"""

    label: str
    samples_map: Dict[str, np.ndarray]

    def __hash__(self):
        samples_tup = tuple(
            str(sample) for sample in self.samples_map.values()
        ) + tuple(self.label)
        return hash(samples_tup)

    def __eq__(self, other: SamplesSignature):
        return (
            self.label == other.label
            and self.samples_map.keys() == other.samples_map.keys()
            and all(
                np.array_equal(self.samples_map[key], other.samples_map[key])
                for key in self.samples_map
            )
        )


@dataclass(unsafe_hash=True)
class WaveformSignature:
    """Signature of a waveform as stored in waveform memory.

    The underlying promise is that two waveforms with the same signature are
    guaranteed to resolve to the same samples, so we need only store one of them and
    can use them interchangeably.

    Alternatively, after compression it makes no sense to associate a WaveformSignature to
    PulseSignatures anymore, since compression can cause a single pulse to be replaced with
    any number of PlayWave and PlayHold statements. In this case, a WaveformSignature is best
    understood as a collection of samples, with the implied promise that equal samples are only
    uploaded to the device once.
    """

    length: int
    pulses: Optional[Tuple[PulseSignature, ...]]
    samples: Optional[SamplesSignature] = field(default=None)

    def signature_string(self):
        retval = "p_" + str(self.length).zfill(4)
        if self.pulses is not None:
            for pulse_entry in self.pulses:
                retval += "_"
                retval += pulse_entry.pulse or ""
                for key, separator, scale, fill in (
                    ("start", "_", 1, 2),
                    ("amplitude", "_a", 1e9, 10),
                    ("length", "_l", 1, 3),
                    ("baseband_phase", "_bb", 1, 7),
                    ("channel", "_c", 1, 0),
                    ("sub_channel", "_sc", 1, 0),
                    ("phase", "_ap", 1, 0),
                ):
                    value = getattr(pulse_entry, key)
                    if value is not None:
                        sign = ""
                        if value < 0:
                            sign = "m"
                        new_part = (
                            separator
                            + sign
                            + str(round(np.abs(scale * value))).zfill(fill)
                        )
                        if len(retval + new_part) > 56:
                            break
                        retval += new_part
                else:  # allow inner break to exit outer loop
                    continue
                break
        if self.samples is not None:
            sample_to_shorthand = {
                "samples_i": "si",
                "samples_q": "sq",
                "samples_marker1": "m1",
                "samples_marker2": "m2",
            }
            for sample_name in self.samples.samples_map.keys():
                new_part = (
                    sample_to_shorthand[sample_name]
                    if sample_name in sample_to_shorthand
                    else sample_name
                )
                if len(retval + new_part) > 56:
                    break
                retval += new_part
            retval += f"_{self.samples.label}_"

        retval = string_sanitize(retval)
        hashed_signature = self.stable_hash().hexdigest()[:7]
        retval += "_" + hashed_signature
        return retval

    def stable_hash(self):
        def default(obj):
            if isinstance(obj, frozenset):
                return tuple(sorted(obj))
            if obj.__class__ in (complex, np.complex128):
                return repr(obj)
            raise TypeError

        return hashlib.sha1(
            # leverage that orjson can serialize dataclasses
            orjson.dumps(
                self,
                option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
                default=default,
            )
        )


@dataclass(unsafe_hash=True)
class PlaybackSignature:
    """Signature of the output produced by a single playback command.

    When using the command table, a single waveform may be used by different table
    entries (different playbacks). This structure captures the additional
    information beyond the sampled waveform."""

    waveform: Optional[WaveformSignature]
    hw_oscillator: Optional[str]
    pulse_parameters: Tuple[Tuple[frozenset, ...], ...]
    state: Optional[int] = None
    set_phase: Optional[float] = None
    increment_phase: Optional[float] = None
    set_amplitude: Optional[float] = None
    clear_precompensation: bool = False

    def quantize_phase(self, phase_resolution_range: int):
        """Quantize the phase of all pulses in the waveform.

        For the phase that is baked into the samples, we can quantize to the precision
        given by `phase_resolution_range`. For the phase specified by registers on the
        device (e.g. command table) we quantize to a fixed precision of 32 bits. This
        serves to avoid rounding errors leading to multiple command table entries."""

        PHASE_RESOLUTION_CT = 1 << 32

        for pulse in self.waveform.pulses:
            if pulse.phase is not None:
                pulse.phase = normalize_phase(
                    round(pulse.phase / 2 / math.pi * phase_resolution_range)
                    / phase_resolution_range
                    * 2
                    * math.pi
                )
        if self.set_phase is not None:
            self.set_phase = normalize_phase(
                round(self.set_phase / 2 / math.pi * PHASE_RESOLUTION_CT)
                / PHASE_RESOLUTION_CT
                * 2
                * math.pi
            )
        if self.increment_phase is not None:
            self.increment_phase = normalize_phase(
                round(self.increment_phase / 2 / math.pi * PHASE_RESOLUTION_CT)
                / PHASE_RESOLUTION_CT
                * 2
                * math.pi
            )


def reduce_signature_phase(
    signature: PlaybackSignature,
    use_ct_phase: bool,
    prev_hw_oscillator_phase: Optional[float],
) -> PlaybackSignature:
    """Reduces the phase of the signature.

    Modifies the passed in `signature` object in-place.
    """
    if use_ct_phase:
        this_hw_oscillator_phase = signature.waveform.pulses[-1].baseband_phase or 0.0
        if prev_hw_oscillator_phase is not None:
            increment = (this_hw_oscillator_phase - prev_hw_oscillator_phase) % (
                2 * math.pi
            )
            if increment != 0:
                signature.increment_phase = increment
        else:
            # oscillator phase from previous phase unknown, set it directly instead of
            # incrementing
            signature.set_phase = this_hw_oscillator_phase
        for pulse in signature.waveform.pulses:
            pulse.baseband_phase = (
                (pulse.baseband_phase or 0.0) - this_hw_oscillator_phase
            ) % (2 * math.pi)

    # absorb the baseband phase into the pulse phase (ie the phase baked into the samples)
    for pulse in signature.waveform.pulses:
        if pulse.baseband_phase is not None:
            pulse.phase = (pulse.phase or 0.0) + pulse.baseband_phase
            pulse.baseband_phase = None
        if pulse.oscillator_phase is not None:
            pulse.phase = (pulse.phase or 0.0) + pulse.oscillator_phase
            pulse.oscillator_phase = None

    return signature


def reduce_signature_amplitude(signature: PlaybackSignature) -> PlaybackSignature:
    """Reduces the amplitude of the signature.

    Modifies the passed in `signature` object in-place.

    Absorb the pulse amplitude into the command table. Whenever possible, the
    waveforms will be sampled at unit amplitude, making waveform reuse more likely.
    """
    if len(signature.waveform.pulses) == 0:
        return signature
    signature.set_amplitude = 1.0
    if any(pulse.amplitude is None for pulse in signature.waveform.pulses):
        return signature

    ct_amplitude = max(abs(pulse.amplitude) for pulse in signature.waveform.pulses)
    ct_amplitude = min(ct_amplitude, +1.0)
    if ct_amplitude != 0:
        for pulse in signature.waveform.pulses:
            pulse.amplitude /= ct_amplitude

    signature.set_amplitude = ct_amplitude

    return signature
