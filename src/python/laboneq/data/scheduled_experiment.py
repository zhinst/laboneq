# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal
import numpy as np
from numpy import typing as npt

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
    channel: int = None  # The AWG channel for rf_signals
    needs_conjugate: bool = False  # SHF devices need that for now
    play_pulse_parameters: dict[str, Any] = field(default_factory=dict)
    pulse_pulse_parameters: dict[str, Any] = field(default_factory=dict)

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
    waveforms: dict[str, PulseWaveformMap] = field(default_factory=dict)


COMPLEX_USAGE = "complex_usage"


@dataclass
class ParameterPhaseIncrementMap:
    entries: list[CommandTableMapEntry | Literal[COMPLEX_USAGE]] = field(
        default_factory=list
    )


@dataclass
class CommandTableMapEntry:
    ct_ref: str
    ct_index: int


class CompilerArtifact:
    pass


@dataclass
class WeightInfo:
    id: str
    downsampling_factor: int | None


SignalWeights = list[WeightInfo]
AwgWeights = dict[str, SignalWeights]


@dataclass
class CodegenWaveform:
    samples: npt.NDArray[Any]
    hold_start: int | None = None
    hold_length: int | None = None
    downsampling_factor: int | None = None

    def __eq__(self, other) -> bool:
        if other is self:
            return True
        if not isinstance(other, CodegenWaveform):
            return False
        return (self.hold_start, self.hold_length, self.downsampling_factor) == (
            other.hold_start,
            other.hold_length,
            other.downsampling_factor,
        ) and np.allclose(self.samples, other.samples)


@dataclass
class ArtifactsCodegen(CompilerArtifact):
    # The SeqC source code, per device.
    src: list[dict[str, str]] | None = None

    # The waveforms that will be uploaded to the devices.
    waves: dict[str, CodegenWaveform] = field(default_factory=dict)

    # Device ID -> True if requires long readout
    requires_long_readout: dict[str, list[str]] = field(default_factory=dict)

    # Data structure for storing the indices or filenames by which the waveforms are
    # referred to during and after upload.
    wave_indices: list[dict[str, Any]] | None = None

    # Data structure for storing the command table data
    command_tables: list[dict[str, Any]] = field(default_factory=list)

    # Data structure for mapping pulses (in the experiment) to waveforms (on the
    # device).
    pulse_map: dict[str, PulseMapEntry] | None = None

    # Data structure mapping pulse parameters for phase increments to command table entries
    parameter_phase_increment_map: dict[str, ParameterPhaseIncrementMap] = field(
        default_factory=dict
    )

    # Data structure for referencing the waveforms used as integration kernels.
    integration_weights: dict[str, AwgWeights] = field(default_factory=dict)


@dataclass
class ScheduledExperiment:
    uid: str | None = None

    #: Instructions to the controller for running the experiment.
    recipe: Recipe | None = None

    #: Compiler artifacts specific to backend(s)
    artifacts: CompilerArtifact | None = None

    def __getattr__(self, attr):
        return getattr(self.artifacts, attr)  # @IgnoreException

    def __copy__(self):
        new_artifacts = copy.copy(self.artifacts)
        new_scheduled_experiment = self.__class__(
            uid=self.uid,
            artifacts=new_artifacts,
            schedule=self.schedule,
            execution=self.execution,
            compilation_job_hash=self.compilation_job_hash,
            experiment_hash=self.experiment_hash,
        )
        return new_scheduled_experiment

    #: list of events as scheduled by the compiler.
    schedule: dict[str, Any] | None = None

    #: Experiment execution model
    execution: Any = None  # TODO(2K): 'Statement' type after refactoring

    compilation_job_hash: str | None = None
    experiment_hash: str | None = None

    def __eq__(self, other):
        if other is self:
            return True

        if not isinstance(other, ScheduledExperiment):
            return NotImplemented

        if len(other.waves) != len(self.waves):
            return False

        return (
            other.uid,
            other.artifacts,
            other.compilation_job_hash,
            other.experiment_hash,
        ) == (
            self.uid,
            other.artifacts,
            self.compilation_job_hash,
            self.experiment_hash,
        ) and dicts_equal(
            {n: w.samples for n, w in other.waves.items()},
            {n: w.samples for n, w in self.waves.items()},
        )
