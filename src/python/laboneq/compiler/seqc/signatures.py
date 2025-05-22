# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
from orjson import orjson

from laboneq.compiler.seqc.utils import normalize_phase
from laboneq.core.utilities.string_sanitize import string_sanitize


@dataclass(unsafe_hash=True)
class PulseSignature:
    """Signature of a single pulse, part of a sampled waveform"""

    # the offset of the pulse in the waveform
    start: int
    # the pulse UID
    pulse: str | None
    # the length of the pulse in samples
    length: int
    # the amplitude of the pulse
    amplitude: float | None
    # the phase of the pulse
    phase: float | None
    # the oscillator phase of the pulse (for SW oscillators)
    oscillator_phase: float | None
    # the oscillator frequency of the pulse (for SW oscillators)
    oscillator_frequency: float | None
    # if present, the pulse increments the HW oscillator phase
    increment_oscillator_phase: float | None
    # the channel of the pulse (for HDAWG)
    channel: Optional[int]
    # the sub-channel of the pulse (for SHFQA)
    sub_channel: Optional[int]
    # Pulse parameters ID
    id_pulse_params: int | None
    # markers played during this pulse
    markers: Any
    # the preferred amplitude register for this pulse, will be aggregated into PlaybackSignature
    preferred_amplitude_register: Optional[int]
    # if this pulse increments the oscillator phase, this field indicates the name of
    # the sweep parameters that determine the increment (if applicable)
    incr_phase_params: tuple[str | None, ...]


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

    def to_frozen(self) -> FrozenWaveformSignature:
        """Create a frozen signature of the waveform.

        This is required as the `WaveformSignature` itself can be modified
        and attaching the signature string into it, would mess with it's hash
        which is used for lookups.
        """
        return FrozenWaveformSignature(
            signature=self.signature_string(),
            length=self.length,
            pulses=self.pulses,
            hash_=hash(self),
            samples=self.samples,
        )

    def signature_string(self) -> str:
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
        hashed_signature = self._stable_hash().hexdigest()[:7]
        retval += "_" + hashed_signature
        return retval

    def _stable_hash(self):
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


@dataclass
class FrozenWaveformSignature(WaveformSignature):
    """Frozen waveform signature.

    Once the signature of a waveform has been fully established, we render it immutable.
    """

    hash_: str | None = None
    signature: str | None = None

    def __hash__(self):
        return self.hash_

    def __eq__(self, other):
        return (
            self.length == other.length
            and self.pulses == other.pulses
            and self.samples == other.samples
        )

    def signature_string(self) -> str:
        return self.signature


@dataclass(frozen=True)
class HWOscillator:
    osc_id: str
    osc_index: int

    @staticmethod
    def make(osc_id: str | None, osc_index: int) -> HWOscillator | None:
        if osc_id is None:
            return None
        return HWOscillator(osc_id=osc_id, osc_index=osc_index)


@dataclass(unsafe_hash=True)
class PlaybackSignature:
    """Signature of the output produced by a single playback command.

    When using the command table, a single waveform may be used by different table
    entries (different playbacks). This structure captures the additional
    information beyond the sampled waveform."""

    waveform: Optional[WaveformSignature]
    hw_oscillator: HWOscillator | None
    state: Optional[int] = None
    set_phase: float | None = None
    increment_phase: float | None = None
    # A collection of the pulse parameters that drive the phase increment. If some phase
    # increment is not associated with a parameter, this is indicated by the special value `None`
    increment_phase_params: tuple[None | str, ...] = field(default_factory=tuple)
    set_amplitude: float | None = None
    increment_amplitude: float | None = None
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

        if self.waveform is not None:
            for pulse in self.waveform.pulses:
                if pulse.phase is not None:
                    pulse.phase = normalize_phase(
                        round(pulse.phase * phase_resolution_range)
                        / phase_resolution_range
                    )
        if self.set_phase is not None:
            self.set_phase = normalize_phase(
                round(self.set_phase * PHASE_RESOLUTION_CT) / PHASE_RESOLUTION_CT
            )
        if self.increment_phase is not None:
            self.increment_phase = normalize_phase(
                round(self.increment_phase * PHASE_RESOLUTION_CT) / PHASE_RESOLUTION_CT
            )
