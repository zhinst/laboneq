# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from numpy import typing as npt

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.types.enums.wave_type import WaveType
from laboneq.core.types.numpy_support import NumPyArray
from laboneq.core.validators import dicts_equal
from laboneq.data import EnumReprMixin
from laboneq.data.awg_info import AwgKey
from laboneq.data.recipe import Recipe
from laboneq.executor.executor import Statement


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
    # The SeqC program, per device.
    src: list[dict[str, str | bytes]] | None = None

    # The waveforms that will be uploaded to the devices.
    waves: dict[str, CodegenWaveform] = field(default_factory=dict)

    # Device ID -> True if requires long readout
    requires_long_readout: dict[str, list[str]] = field(default_factory=dict)

    # Data structure for storing the indices or filenames by which the waveforms are
    # referred to during and after upload.
    wave_indices: list[dict[str, str | dict[str, tuple[int, WaveType]]]] | None = None

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
class HandleResultShape:
    signal: str
    shape: tuple[int, ...]
    axis_names: list[str | list[str]]
    axis_values: list[NumPyArray | list[NumPyArray]]
    chunked_axis_index: int | None
    # Maps axis to sorted indices of rows along that axis that correspond to this result
    # e.g. if shape = (3, 5, 7), and mask = {2: [6, 8]}, then this result fills the subarray [:, :, [6, 8]]
    match_case_mask: dict[int, list[int]] | None

    def __eq__(self, other):
        if id(self) == id(other):
            return True
        if not isinstance(other, HandleResultShape):
            return False

        return (
            self.signal,
            self.shape,
            self.axis_names,
            self.chunked_axis_index,
            self.match_case_mask,
        ) == (
            self.signal,
            other.shape,
            other.axis_names,
            other.chunked_axis_index,
            other.match_case_mask,
        )


@dataclass(frozen=True)
class ResultSource:
    device_id: str
    awg_id: int | str
    integrator_idx: int | None  # None for RAW acquisition


@dataclass
class RtLoopProperties:
    uid: str
    acquisition_type: AcquisitionType
    averaging_mode: AveragingMode
    shots: int
    chunk_count: int | None


@dataclass(frozen=True)
class ResultShapeInfo:
    shapes: dict[str, HandleResultShape]
    result_handle_maps: dict[ResultSource, list[set[str]]]
    result_lengths: dict[AwgKey, int]


@dataclass
class ScheduledExperiment:
    device_setup_fingerprint: str

    #: Instructions to the controller for running the experiment.
    recipe: Recipe

    #: Compiler artifacts specific to backend(s)
    artifacts: CompilerArtifact

    #: list of events as scheduled by the compiler.
    schedule: dict[str, Any] | None

    #: Experiment execution model
    execution: Statement

    rt_loop_properties: RtLoopProperties

    result_shape_info: ResultShapeInfo

    def __getattr__(self, attr):
        return getattr(self.artifacts, attr)  # @IgnoreException

    def __eq__(self, other):
        if other is self:
            return True

        if not isinstance(other, ScheduledExperiment):
            return NotImplemented

        if len(other.waves) != len(self.waves):
            return False

        return (
            other.device_setup_fingerprint,
            other.artifacts,
            other.result_shape_info,
        ) == (
            self.device_setup_fingerprint,
            self.artifacts,
            self.result_shape_info,
        ) and dicts_equal(
            {n: w.samples for n, w in other.waves.items()},
            {n: w.samples for n, w in self.waves.items()},
        )
