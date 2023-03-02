# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np
from orjson import orjson

from laboneq.compiler.code_generator.seq_c_generator import string_sanitize


@dataclass(frozen=True)
class PulseSignature:
    """Signature of a single pulse, part of a sampled waveform"""

    start: int
    pulse: str
    length: int
    amplitude: Optional[float]
    phase: Optional[int]
    oscillator_phase: Optional[float]
    oscillator_frequency: Optional[float]
    baseband_phase: Optional[float]  # todo: rename to `persistent_phase`
    channel: Optional[int]
    sub_channel: Optional[int]
    pulse_parameters: FrozenSet[Tuple[str, Any]]
    markers: Any


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
                        separator + sign + str(abs(round(scale * value))).zfill(fill)
                    )
                    if len(retval + new_part) > 56:
                        break
                    retval += new_part
            else:  # allow inner break to exit outer loop
                continue
            break

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


@dataclass(frozen=True)
class PlaybackSignature:
    """Signature of the output produced by a single playback command.

    When using the command table, a single waveform may be used by different table
    entries (different playbacks). This structure captures the additional
    information beyond the sampled waveform."""

    waveform: WaveformSignature
    hw_oscillator: Optional[str]
    pulse_parameters: List[Tuple[Dict, Dict]]
    state: Optional[int] = None
