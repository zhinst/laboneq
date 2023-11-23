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

    # the offset of the pulse in the waveform
    start: int
    # the pulse function
    pulse: str
    # the length of the pulse in samples
    length: int
    # the amplitude of the pulse
    amplitude: Optional[float]
    # the phase of the pulse
    phase: Optional[float]
    # the oscillator phase of the pulse (for SW oscillators)
    oscillator_phase: Optional[float]
    # the oscillator frequency of the pulse (for SW oscillators)
    oscillator_frequency: Optional[float]
    # if present, the pulse increments the HW oscillator phase
    increment_oscillator_phase: Optional[float]
    # the channel of the pulse (for HDAWG)
    channel: Optional[int]
    # the sub-channel of the pulse (for SHFQA)
    sub_channel: Optional[int]
    # additional user pulse parameters
    pulse_parameters: FrozenSet[Tuple[str, str]]
    # markers played during this pulse
    markers: Any
    # the preferred amplitude register for this pulse, will be aggregated into PlaybackSignature
    preferred_amplitude_register: Optional[int]


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
                    ("amplitude", "_a", 1e3, 4),
                    ("length", "_l", 1, 3),
                    ("increment_oscillator_phase", "_ip", 1e3 / 2 / math.pi, 4),
                    ("channel", "_c", 1, 0),
                    ("sub_channel", "_sc", 1, 0),
                    ("phase", "_ap", 1e3 / 2 / math.pi, 4),
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
    hw_oscillator: str | None
    pulse_parameters: Tuple[Tuple[frozenset, ...], ...]
    state: Optional[int] = None
    set_phase: Optional[float] = None
    increment_phase: Optional[float] = None
    set_amplitude: Optional[float] = None
    increment_amplitude: Optional[float] = None
    amplitude_register: int = 0
    clear_precompensation: bool = False

    def quantize_phase(self, phase_resolution_range: int):
        """Quantize the phase of all pulses in the waveform.

        For the phase that is baked into the samples, we can quantize to the precision
        given by `phase_resolution_range`. For the phase specified by registers on the
        device (e.g. command table) we quantize to a fixed precision of 24 bits. This
        serves to avoid rounding errors leading to multiple command table entries."""

        PHASE_RESOLUTION_CT = (1 << 24) / (2 * math.pi)
        phase_resolution_range /= 2 * math.pi

        for pulse in self.waveform.pulses:
            if pulse.phase is not None:
                pulse.phase = normalize_phase(
                    round(pulse.phase * phase_resolution_range) / phase_resolution_range
                )
        if self.set_phase is not None:
            self.set_phase = normalize_phase(
                round(self.set_phase * PHASE_RESOLUTION_CT) / PHASE_RESOLUTION_CT
            )
        if self.increment_phase is not None:
            self.increment_phase = normalize_phase(
                round(self.increment_phase * PHASE_RESOLUTION_CT) / PHASE_RESOLUTION_CT
            )

    def quantize_amplitude(self, amplitude_resolution_range: int):
        """Quantize the amplitude of all pulses in the waveform.

        For the amplitude that is baked into the samples, we can quantize to the precision
        given by `amplitude_resolution_range`. For the amplitude specified by registers on the
        device (e.g. command table) we quantize to a fixed precision of 18 bits. This
        serves to avoid rounding errors leading to multiple command table entries."""

        AMPLITUDE_RESOLUTION_CT = 1 << 18

        if self.waveform is not None:
            for pulse in self.waveform.pulses:
                if pulse.amplitude is not None:
                    pulse.amplitude = (
                        round(pulse.amplitude * amplitude_resolution_range)
                        / amplitude_resolution_range
                    )
        if self.set_amplitude is not None:
            self.set_amplitude = (
                round(self.set_amplitude * AMPLITUDE_RESOLUTION_CT)
                / AMPLITUDE_RESOLUTION_CT
            )
        if self.increment_amplitude is not None:
            self.increment_amplitude = (
                round(self.increment_amplitude * AMPLITUDE_RESOLUTION_CT)
                / AMPLITUDE_RESOLUTION_CT
            )

            if self.increment_amplitude == 0 and not self.amplitude_register:
                self.increment_amplitude = None


def reduce_signature_phase(
    signature: PlaybackSignature,
    use_ct_phase: bool,
    after_phase_reset: bool = False,
) -> PlaybackSignature:
    """Reduces the phase of the signature.

    Modifies the passed in `signature` object in-place.
    """
    total_phase_increment = sum(
        pulse.increment_oscillator_phase or 0.0 for pulse in signature.waveform.pulses
    )
    if total_phase_increment:
        assert use_ct_phase, "cannot increment oscillator phase w/o command table"

        if after_phase_reset:
            signature.set_phase = total_phase_increment
        else:
            signature.increment_phase = total_phase_increment

    # absorb the partial phase increment into the pulse phase (ie the phase baked into the samples)
    running_increment = 0
    for pulse in signature.waveform.pulses:
        running_increment += pulse.increment_oscillator_phase or 0.0
        pulse.increment_oscillator_phase = None

        if running_increment - total_phase_increment:
            pulse.oscillator_phase = (
                (pulse.oscillator_phase or 0.0)
                + running_increment
                - total_phase_increment
            )

        # todo: this seems wrong - it will break pulse replacement
        if pulse.oscillator_phase is not None:
            pulse.phase = (pulse.phase or 0.0) + pulse.oscillator_phase
            pulse.oscillator_phase = None

    return signature


def reduce_signature_amplitude_register(
    signature: PlaybackSignature,
) -> PlaybackSignature:
    """Determine the amplitude register

    Modifies the passed `signature` object in-place.

    Aggregate the preferred amplitude register of the individual pulses into the base
    `PlaybackSignature.amplitude_register`.
    Any waveforms that include two or more pulses that prefer different registers, the
    result will also fall-back to register 0.
    """
    if signature.waveform is None:
        return signature

    requested_registers = set(
        p.preferred_amplitude_register for p in signature.waveform.pulses
    )
    register = 0
    try:
        [register] = requested_registers
    except ValueError:
        # too many registers -> fall-back to R0
        pass

    # set the base register, and clear the per-pulse register
    for pulse in signature.waveform.pulses:
        pulse.preferred_amplitude_register = None
    signature.amplitude_register = register

    return signature


def _split_complex_amplitude(signature):
    if signature.waveform is None:
        return signature
    for pulse in signature.waveform.pulses:
        if isinstance(pulse.amplitude, complex):
            theta = float(np.angle(pulse.amplitude))
            pulse.phase = normalize_phase((pulse.phase or 0.0) - theta)
            pulse.amplitude = abs(pulse.amplitude)

    return signature


def _aggregate_ct_amplitude(signature) -> PlaybackSignature:
    if signature.set_amplitude is None:
        signature.set_amplitude = 1.0

    if signature.waveform is not None and len(signature.waveform.pulses) > 0:
        if any(pulse.amplitude is None for pulse in signature.waveform.pulses):
            return signature

        ct_amplitude = max(abs(pulse.amplitude) for pulse in signature.waveform.pulses)
        ct_amplitude = min(ct_amplitude, +1.0)
        if ct_amplitude != 0:
            for pulse in signature.waveform.pulses:
                pulse.amplitude /= ct_amplitude

        signature.set_amplitude = ct_amplitude
    return signature


def _make_ct_amplitude_incremental(
    signature, previous_amplitude_registers: list[float] | None
) -> PlaybackSignature:
    if signature.state is not None:
        # For branches, we _must_ emit the same signature every time, and cannot depend
        # what came before. So using the increment is not valid.
        return signature
    if signature.amplitude_register > 0:
        previous_amplitude = None
        if previous_amplitude_registers is not None:
            previous_amplitude = previous_amplitude_registers[
                signature.amplitude_register
            ]
        if previous_amplitude is not None:
            signature.increment_amplitude = signature.set_amplitude - previous_amplitude
            signature.set_amplitude = None

    return signature


def reduce_signature_amplitude(
    signature: PlaybackSignature,
    use_command_table: bool = True,
    previous_amplitude_registers: list[float] | None = None,
) -> PlaybackSignature:
    """Reduce (simplify) the amplitude of the signature.

    Modifies the passed `signature` object in-place.

    Split complex amplitudes into a real amplitude and a phase.

    Absorb the pulse amplitude into the command table. Whenever possible, the
    waveforms will be sampled at unit amplitude, making waveform reuse more likely.

    If the current value of the amplitude register is known, the new amplitude is
    expressed as an _increment_ of the old value. This way, amplitude sweeps can be
    modelled with very few command table entries.

    Args:
        signature: The signature to mutate
        use_command_table: Whether to lump the waveform's amplitude into the command table
        previous_amplitude_registers: The current values of the amplitude registers
          These values are not mutated. To express an unknown amplitude register, pass
          `None` in the relevant element. If the entire register file is `None`, all the
          values are assumed unknown.
    """
    signature = _split_complex_amplitude(signature)

    if use_command_table:
        signature = _aggregate_ct_amplitude(signature)

        # Next, we attempt to make the amplitude relative to the previous value.
        # The idea is that if there is a linear sweep, the increment is constant
        # (especially if the register is reserved for a sweep parameter). The same command
        # table entry can then be reused for every step of the sweep.
        signature = _make_ct_amplitude_incremental(
            signature, previous_amplitude_registers
        )

    return signature
