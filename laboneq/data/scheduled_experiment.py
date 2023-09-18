# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from laboneq.core.validators import dicts_equal
from laboneq.data import EnumReprMixin
from laboneq.data.recipe import Recipe


class MixerType(EnumReprMixin, Enum):
    #: Mixer performs full complex modulation
    IQ = "IQ"
    #: Mixer only performs envelope modulation (UHFQA-style)
    UHFQA_ENVELOPE = "UHFQA_ENVELOPE"


@dataclass
class PulseInstance:
    offset_samples: int
    amplitude: float = None  # instance (final) amplitude
    length: float = None  # instance (final) length
    iq_phase: float = None
    modulation_frequency: float = None
    modulation_phase: float = None
    channel: int = None  # The AWG channel for rf_signals
    needs_conjugate: bool = False  # SHF devices need that for now
    play_pulse_parameters: dict[str, Any] = field(default_factory=dict)
    pulse_pulse_parameters: dict[str, Any] = field(default_factory=dict)

    # uid of pulses that this instance overlaps with
    overlaps: list[str] = None
    has_marker1: bool = False
    has_marker2: bool = False
    can_compress: bool = False


@dataclass
class PulseWaveformMap:
    """Data structure to store mappings between the given pulse and an AWG waveform."""

    sampling_rate: float
    length_samples: int
    signal_type: str
    # UHFQA's HW modulation is not an IQ mixer. None for flux pulses etc.
    mixer_type: MixerType | None = None
    instances: list[PulseInstance] = field(default_factory=list)


@dataclass
class PulseMapEntry:
    """Data structure to store the :py:class:`PulseWaveformMap` of each AWG waveform."""

    # key: waveform signature string
    #: A mapping of signals to :py:class:`PulseWaveformMap`
    waveforms: dict[str, PulseWaveformMap] = field(default_factory=dict)


@dataclass
class ScheduledExperiment:
    uid: str = None

    #: Instructions to the controller for running the experiment.
    recipe: Recipe = None

    #: The SeqC source code, per device.
    src: list[dict[str, str]] = None

    #: The waveforms that will be uploaded to the devices.
    waves: list[dict[str, Any]] = None

    #: Data structure for storing the indices or filenames by which the waveforms are
    #: referred to during and after upload.
    wave_indices: list[dict[str, Any]] = None

    #: Data structure for storing the command table data
    command_tables: list[dict[str, Any]] = field(default_factory=list)

    #: list of events as scheduled by the compiler.
    schedule: dict[str, Any] = None

    #: Data structure for mapping pulses (in the experiment) to waveforms (on the
    #: device).
    pulse_map: dict[str, PulseMapEntry] = None

    #: Experiment execution model
    execution: Any = None  # TODO(2K): 'Statement' type after refactoring

    compilation_job_hash: str = None
    experiment_hash: str = None

    def __eq__(self, other):
        if other is self:
            return True

        if type(other) is not ScheduledExperiment:
            return NotImplemented

        if len(other.waves) != len(self.waves):
            return False

        return (
            other.uid,
            other.recipe,
            other.src,
            other.wave_indices,
            other.command_tables,
            other.schedule,
            other.pulse_map,
            other.execution,
            other.compilation_job_hash,
            other.experiment_hash,
        ) == (
            self.uid,
            self.recipe,
            self.src,
            self.wave_indices,
            self.command_tables,
            self.schedule,
            self.pulse_map,
            self.execution,
            self.compilation_job_hash,
            self.experiment_hash,
        ) and dicts_equal(
            other.waves, self.waves
        )
