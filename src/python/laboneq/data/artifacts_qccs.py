# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from typing_extensions import TypeAlias

from laboneq.data.scheduled_experiment import CompilerArtifact

if TYPE_CHECKING:
    from numpy import typing as npt

    from laboneq.core.types.enums.awg_signal_type import AWGSignalType
    from laboneq.core.types.enums.mixer_type import MixerType
    from laboneq.core.types.enums.port_mode import PortMode
    from laboneq.core.types.enums.wave_type import WaveType
    from laboneq.data.calibration import CancellationSource
    from laboneq.data.nt_step_key import NtStepKey

ParameterUID: TypeAlias = str


@dataclass
class Gains:
    diagonal: float | ParameterUID
    off_diagonal: float | ParameterUID


@dataclass
class RoutedOutput:
    """Output route of Output Router and Adder (RTR)."""

    from_channel: int
    amplitude: float | ParameterUID
    phase: float | ParameterUID


@dataclass
class IO:
    channel: int
    enable: bool | None = None
    modulation: bool | None = None
    offset: float | None | ParameterUID = None
    gains: Gains | None = None
    range: float | None = None
    range_unit: str | None = None
    precompensation: dict[str, dict | list | None] | None = None
    lo_frequency: float | str | None = None
    port_mode: PortMode | None = None
    port_delay: float | str | None = None
    scheduler_port_delay: float = 0.0
    marker_mode: str | None = None
    amplitude: float | str | None = None
    routed_outputs: list[RoutedOutput] = field(default_factory=list)
    enable_output_mute: bool = False


@dataclass
class AWG:
    awg: int
    signal_type: AWGSignalType
    signals: set[str] = field(default_factory=set)

    # receiver (SG instruments)
    source_feedback_register: int | Literal["local"] | None = None
    codeword_bitshift: int | None = None
    codeword_bitmask: int | None = None
    feedback_register_index_select: int | None = None
    command_table_match_offset: int | None = None

    # transmitter (QA instruments)
    # TODO(2K): This value is not used by controller, but used in tests. Consider removing it.
    target_feedback_register: int | Literal["local"] | None = None

    # Result length, `None` if the AWG has no acquisitions.
    result_length: int | None = None


@dataclass
class Measurement:
    length: int
    channel: int = 0


@dataclass
class Config:
    lead_delay: float = 0.0
    sampling_rate: float | None = None


@dataclass
class PPChannel:
    """Amplifier pump (SHFPPC) settings for a single channel."""

    channel: int
    pump_on: bool
    cancellation_on: bool
    cancellation_source: CancellationSource
    cancellation_source_frequency: float | ParameterUID | None
    alc_on: bool
    pump_filter_on: bool
    probe_on: bool
    pump_frequency: float | ParameterUID | None
    pump_power: float | ParameterUID | None
    probe_frequency: float | ParameterUID | None
    probe_power: float | ParameterUID | None
    cancellation_phase: float | ParameterUID | None
    cancellation_attenuation: float | ParameterUID | None
    # JSON string with sweep config, `None` if not swept.
    sweep_config: str | None


@dataclass
class Initialization:
    device_uid: str
    device_type: str | None = None
    config: Config = field(default_factory=Config)
    awgs: list[AWG] = field(default_factory=list)
    outputs: list[IO] = field(default_factory=list)
    inputs: list[IO] = field(default_factory=list)
    measurements: list[Measurement] = field(default_factory=list)
    ppchannels: list[PPChannel] = field(default_factory=list)


@dataclass
class OscillatorParam:
    id: str
    device_id: str
    channel: int
    signal_id: str
    allocated_index: int
    frequency: float | None = None
    param: str | None = None


@dataclass
class IntegratorAllocation:
    signal_id: str
    device_id: str
    awg: int
    channels: list[int]
    kernel_count: int
    thresholds: list[float] = field(default_factory=lambda: [0.0])


@dataclass
class AcquireLength:
    signal_id: str
    acquire_length: int


@dataclass
class RealtimeExecutionInit:
    device_id: str
    awg_index: int
    program_ref: str
    nt_step: NtStepKey
    wave_indices_ref: str | None = None
    kernel_indices_ref: str | None = None


@dataclass
class PulseInstance:
    offset_samples: int
    amplitude: float | None = None  # instance (final) amplitude
    length: float | None = None  # instance (final) length
    iq_phase: float | None = None
    modulation_frequency: float | None = None
    channel: int | None = None  # The AWG channel for rf_signals
    needs_conjugate: bool = False  # SHF devices need that for now
    play_pulse_parameters: dict[str, Any] = field(default_factory=dict)
    pulse_pulse_parameters: dict[str, Any] = field(default_factory=dict)

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


@dataclass
class WeightInfo:
    id: str
    integration_units: list[int]
    downsampling_factor: int | None


AwgWeights = list[WeightInfo]


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


@dataclass(frozen=True)
class ResultSource:
    device_id: str
    awg_id: int
    # The first integration unit allocated to the signal on this AWG. Acts as a stable
    # per-signal routing key; unique by construction (see `allocate_integration_units`
    # in the Rust code generator, which hands out non-overlapping unit ranges per AWG).
    # None for RAW acquisition, where results are per physical port rather than per integrator.
    integrator_idx: int | None


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

    # The handles of each result index, per result source. For a single-shot
    # experiment, the indices cover one shot. A source has one entry per
    # acquisition round of its AWG. An empty set marks a round in which the
    # source's signal does not acquire.
    result_handle_maps: dict[ResultSource, list[set[str]]] = field(default_factory=dict)

    # QCCS device/AWG initialization records, per device.
    initializations: list[Initialization] = field(default_factory=list)

    # Per-AWG initialisation records for the controller's near-time execution steps.
    realtime_execution_init: list[RealtimeExecutionInit] = field(default_factory=list)

    # Hardware oscillator allocations.
    oscillator_params: list[OscillatorParam] = field(default_factory=list)

    # Integration unit allocations for acquired signals.
    integrator_allocations: list[IntegratorAllocation] = field(default_factory=list)

    # Acquire lengths per signal.
    acquire_lengths: list[AcquireLength] = field(default_factory=list)
