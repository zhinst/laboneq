# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


from laboneq.compiler.seqc.utils import normalize_phase
from laboneq._rust.codegenerator import WaveformSignature


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
    reset_phase: bool = False
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
        if self.increment_phase is not None:
            self.increment_phase = normalize_phase(
                round(self.increment_phase * PHASE_RESOLUTION_CT) / PHASE_RESOLUTION_CT
            )
