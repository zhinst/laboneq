# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import sys

from dataclasses import dataclass
from typing import Optional, FrozenSet, Tuple, Any

from laboneq.compiler.code_generator.seq_c_generator import string_sanitize


@dataclass(frozen=True)
class PulseSignature:
    """Signature of a single pulse, part of a sampled waveform"""

    start: int
    end: int
    pulse: str
    pulse_samples: int
    amplitude: Optional[float]
    phase: Optional[int]
    oscillator_phase: Optional[float]
    baseband_phase: Optional[float]  # todo: rename to `persistent_phase`
    channel: Optional[int]
    sub_channel: Optional[int]
    pulse_parameters: FrozenSet[Tuple[str, Any]]


@dataclass(frozen=True)
class WaveformSignature:
    """Signature of a waveform as stored in waveform memory.

    The underlying promise is that two waveforms with the same signature are
    guaranteed to resolve to the same samples, so we need only store one of them and
    can use them interchangeably."""

    length: int
    pulses: Tuple[PulseSignature]

    def signature_string(self):
        retval = "p_" + str(self.length).zfill(4)
        for pulse_entry in self.pulses:
            retval += "_"
            retval += pulse_entry.pulse
            for key, separator, scale, fill in (
                ("start", "_", 1, 2),
                ("amplitude", "_a", 1e9, 10),
                ("pulse_samples", "_l", 1, 3),
                ("oscillator_phase", "_ph", 1, 7),
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
                    retval += (
                        separator + sign + str(abs(round(scale * value))).zfill(fill)
                    )
            # Simplified approach with hash of all parameters.
            # Can be expanded to individual params, but it will
            # require handling of unknown types, ranges and
            # scales of the parameters.
            pulse_parameters = pulse_entry.pulse_parameters
            if pulse_parameters is not None and len(pulse_parameters) > 0:
                pp_hash = hash(pulse_parameters)
                if pp_hash < 0:
                    pp_hash += 1 << sys.hash_info.width
                retval += f"_pp{pp_hash:016X}"

        retval = string_sanitize(retval)

        long_signature = None
        if len(retval) > 64:
            hashed_signature = hashlib.md5(retval.encode()).hexdigest()
            long_signature = retval
            retval = hashed_signature

        return retval, long_signature


@dataclass(frozen=True)
class PlaybackSignature:
    """Signature of the output produced by a single playback command.

    When using the command table, a single waveform may be used by different table
    entries (different playbacks). This structure captures the additional
    information beyond the sampled waveform."""

    waveform: WaveformSignature
    hw_oscillator: Optional[str]
