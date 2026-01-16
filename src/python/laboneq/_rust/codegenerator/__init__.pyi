# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy.typing as npt

from laboneq.compiler.feedback_router.feedback_router import FeedbackRegisterLayout
from laboneq.compiler.ir import IRTree
from laboneq.compiler.seqc.waveform_sampler import SampledWaveformSignature
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.data.awg_info import AWGInfo
from laboneq.data.compilation_job import Marker

class PulseSignature:
    start: int
    length: int
    pulse: str | None
    amplitude: float | None
    phase: float | None
    oscillator_frequency: float | None
    channel: int | None
    sub_channel: int | None
    id_pulse_params: int | None
    markers: list[Marker]

class PlaySamples:
    def __init__(
        self,
        offset: int,
        length: int,
        uid: int,
        label: str,
        has_i: bool,
        has_q: bool | None,
        has_marker1: bool | None,
        has_marker2: bool | None,
        signature: SampledWaveformSignature,
    ) -> None: ...

class PlayHold:
    def __init__(self, offset: int, length: int) -> None: ...

class WaveformSignature:
    length: int
    pulses: list[PulseSignature]

    def is_playzero(self) -> bool:
        """Check if the waveform signature represents a play zero waveform."""

    def signature_string(self) -> str:
        """Generate a string representation of the waveform signature."""

class SampledWaveform:
    signals: set[str]
    signature: SampledWaveformSignature
    signature_string: str

class IntegrationWeight:
    basename: str
    samples_i: npt.ArrayLike
    samples_q: npt.ArrayLike
    downsampling_factor: int | None
    signals: set[str]

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

class AwgCodeGenerationResult:
    seqc: str
    wave_indices: list[tuple[str, tuple[int, str]]]
    command_table: dict[str, object] | None
    shf_sweeper_config: dict[str, object] | None
    sampled_waveforms: list[SampledWaveform]
    integration_weights: list[IntegrationWeight]
    # Signal delays in seconds to be applied to the signal
    signal_delays: dict[str, float] = {}
    # Signal integration lengths in seconds
    # This is a mapping from signal name to SignalIntegrationInfo
    integration_lengths: dict[str, SignalIntegrationInfo] = {}
    parameter_phase_increment_map: dict[str, list[int]] | None
    feedback_register_config: FeedbackRegisterConfig

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
    total_execution_time: float = 0.0
    result_handle_maps: dict[ResultSource, list[list[str]]] = {}
    measurements: list[Measurement] = []

def generate_code(
    ir: IRTree,
    awgs: list[AWGInfo],
    settings: dict[str, bool | int | float],
    waveform_sampler: object,
    feedback_register_layout: FeedbackRegisterLayout | None,
    acquisition_type: AcquisitionType,
) -> SeqCGenOutput:
    """Generate SeqC code for given AWGs.

    Arguments:
        ir: The IR tree containing the data to be processed.
        awgs: List of target awgs.
        settings: Compiler settings as dictionary.
        waveform_sampler: An instance of `WaveformSampler` for waveform sampling.
        feedback_register_layout: The feedback register layout.
        acquisition_type: The acquisition type.
    """
