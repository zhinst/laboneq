# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum, auto
from typing import Literal

import numpy.typing as npt

from laboneq._rust.compiler import ExperimentIr
from laboneq.compiler.seqc.waveform_sampler import SampledWaveformSignature
from laboneq.data.compilation_job import PulseDef

class SignalType(Enum):
    IQ = auto()
    SINGLE = auto()
    INTEGRATION = auto()

class DeviceType(Enum):
    HDAWG = auto()
    SHFQA = auto()
    SHFSG = auto()
    UHFQA = auto()

class MixerType(Enum):
    IQ = auto()
    UhfqaEnvelope = auto()

class PulseParameters:
    parameters: dict
    pulse_parameters: dict
    play_parameters: dict

class MarkerSamplingDesc:
    marker_selector: str
    enable: bool
    start: float | None
    length: float | None
    pulse: PulseDef | None

class PulseSamplingDesc:
    start: int
    length: int
    pulse: PulseDef
    amplitude: float
    phase: float
    oscillator_frequency: float | None
    channel: int | None
    pulse_parameters: PulseParameters | None
    markers: list[MarkerSamplingDesc]

class WaveformSamplingDesc:
    length: int
    pulses: list[PulseSamplingDesc]

class PlaySamples:
    def __init__(
        self,
        offset: int,
        length: int,
        uid: int,
        label: str,
        has_i: bool,
        has_q: bool,
        has_marker1: bool,
        has_marker2: bool,
        signature: SampledWaveformSignature,
    ) -> None: ...

class PlayHold:
    def __init__(self, offset: int, length: int) -> None: ...

class SampledWaveform:
    signals: set[str]
    signature: SampledWaveformSignature
    signature_string: str

class IntegrationKernel:
    basename: str
    samples_i: npt.ArrayLike
    samples_q: npt.ArrayLike
    downsampling_factor: int | None
    signals: list[str]

class SignalIntegrationInfo:
    is_play: bool
    length: int

class FeedbackRegisterConfig:
    local: bool
    source_feedback_register: int | None
    register_index_select: int | None
    codeword_bitshift: int | None
    codeword_bitmask: int | None
    command_table_offset: int | None
    target_feedback_register: int | None

class ChannelProperties:
    signal: str
    channel: int
    direction: Literal["IN", "OUT"]
    marker_mode: Literal["TRIGGER", "MARKER"] | None
    hw_oscillator_index: int | None
    amplitude: float | str | None  # Can be a float or a parameter name
    voltage_offset: float | str | None  # Can be a float or a parameter name
    gains: Gains | None

class Gains:
    diagonal: float | str  # Can be a float or a parameter name
    off_diagonal: float | str  # Can be a float or a parameter name

class AwgProperties:
    key: tuple[str, int]  # (device UID, AWG index)
    signal_type: Literal["IQ", "SINGLE", "DOUBLE"]

class PpcSweeperConfig:
    ppc_device: str
    ppc_channel: int
    json: str

class SeqCProgram:
    src: str
    sequencer: str
    dev_type: str
    dev_opts: list[str]
    awg_index: int
    sampling_rate: float | None

class IntegrationWeight:
    integration_units: list[int]
    basename: str
    downsampling_factor: int

class AwgCodeGenerationResult:
    awg_properties: AwgProperties
    seqc: SeqCProgram
    wave_indices: list[tuple[str, tuple[int, str]]]
    command_table: str | None
    shf_sweeper_config: PpcSweeperConfig | None
    sampled_waveforms: list[SampledWaveform]
    integration_kernels: list[IntegrationKernel]
    # Signal delays in seconds to be applied to the signal
    signal_delays: dict[str, float]
    # Signal integration lengths in seconds
    # This is a mapping from signal name to SignalIntegrationInfo
    integration_lengths: dict[str, SignalIntegrationInfo]
    parameter_phase_increment_map: dict[str, list[int]] | None
    feedback_register_config: FeedbackRegisterConfig
    channel_properties: list[ChannelProperties]
    integration_weights: list[IntegrationWeight]
    integration_unit_allocations: list[IntegrationUnitAllocation]

class Measurement:
    """Measurement information for a device.

    Attributes:
        device: The device identifier.
        length: The length of the measurement in samples.
        channel: The channel number of the device for the measurement.
    """

    device: str
    length: int
    channel: int

class ResultSource:
    device_id: str
    awg_id: int
    integrator_idx: int | None

class IntegrationUnitAllocation:
    signal: str
    integration_units: list[int]
    kernel_count: int
    thresholds: list[float]

class SeqCGenOutput:
    """Output of the SeqC code generation process.

    Attributes:
        awg_results: List of code generation results for each awg.
        total_execution_time: Total execution time in seconds of the generated code.
        result_handle_maps: For each result source contains a mask that identifies the
                           acquire handles corresponding to incoming stream of data.
        measurements: List of Measurement objects.
    """

    awg_results: list[AwgCodeGenerationResult]
    total_execution_time: float
    result_handle_maps: dict[ResultSource, list[list[str]]]
    measurements: list[Measurement]

def generate_code(
    ir_experiment: ExperimentIr,
) -> SeqCGenOutput:
    """Generate SeqC code for given AWGs.

    Arguments:
        ir_experiment: The Rust lib experiment IR.
    """
