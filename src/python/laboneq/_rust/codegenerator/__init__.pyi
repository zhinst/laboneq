# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Sequence, Literal
from enum import Enum
import numpy.typing as npt
from laboneq.compiler.common.awg_signal_type import AWGSignalType
from laboneq.compiler.common.awg_info import AWGInfo
from laboneq.compiler.common.device_type import DeviceType as DeviceTypePy
from laboneq.compiler.ir import IRTree
from laboneq.data.compilation_job import Marker
from laboneq.compiler.seqc.waveform_sampler import SampledWaveformSignature

class DeviceType(Enum):
    HDAWG = 0
    SHFQA = 1
    SHFSG = 2
    UHFQA = 3

class SignalType(Enum):
    IQ = 0
    SINGLE = 1
    INTEGRATION = 2

class MixerType(Enum):
    IQ = 0
    UhfqaEnvelope = 1

class AwgKey:
    """Unique identifier for an AWG.

    The class `__eq__`, `__hash__` and `__repr__` equals to the Python `AwgKey` class.
    Serialization support is not yet implemented and should be avoided.
    Use the Python `AwgKey` class for serialization if needed.
    """

    device_id = str
    awg_id = int

class WaveIndexTracker:
    def __init__(self) -> None: ...
    def lookup_index_by_wave_id(self, wave_id: str) -> int | None: ...
    def create_index_for_wave(self, wave_id: str, signal_type: str) -> int | None: ...
    def add_numbered_wave(self, wave_id: str, signal_type: str, index: int) -> None: ...
    def wave_indices(self) -> dict[str, list[int | str]]: ...

def seqc_generator_from_device_and_signal_type(
    device_type: str, signal_type: str
) -> SeqCGenerator: ...

class SeqCGenerator:
    def create(self) -> "SeqCGenerator": ...
    def num_statements(self) -> int: ...
    def num_noncomment_statements(self) -> int: ...
    def clear(self) -> None: ...
    def append_statements_from(self, seq_c_generator: "SeqCGenerator") -> None: ...
    def add_comment(self, comment_text: str) -> None: ...
    def add_function_call_statement(
        self,
        name: str,
        args: Sequence[bool | int | float | str] | None = None,
        assign_to: str | None = None,
    ) -> None: ...
    def add_wave_declaration(
        self, wave_id: str, length: int, has_marker1: bool, has_marker2: bool
    ) -> None: ...
    def add_zero_wave_declaration(self, wave_id: str, length: int) -> None: ...
    def add_constant_definition(
        self, name: str, value: bool | int | float | str, comment: str | None = None
    ) -> None: ...
    def estimate_complexity(self) -> int: ...
    def add_repeat(self, num_repeats: int, body: SeqCGenerator) -> None: ...
    def add_do_while(self, condition: str, body: SeqCGenerator) -> None: ...
    def add_if(
        self, conditions: Sequence[str | None], bodies: Sequence[SeqCGenerator]
    ) -> None: ...
    def add_function_def(self, text: str) -> None: ...
    def is_variable_declared(self, variable_name: str) -> bool: ...
    def add_variable_declaration(
        self, variable_name: str, initial_value: bool | int | float | str | None = None
    ) -> None: ...
    def add_variable_assignment(
        self, variable_name: str, value: bool | int | float | str
    ) -> None: ...
    def add_variable_increment(
        self, variable_name: str, value: int | float
    ) -> None: ...
    def add_assign_wave_index_statement(
        self, wave_id: str, wave_index: int, channel: int | None
    ) -> None: ...
    def add_play_wave_statement(self, wave_id: str, channel: int | None) -> None: ...
    def add_command_table_execution(
        self, ct_index: int | str, latency: int | str | None = None, comment: str = ""
    ) -> None: ...
    def add_play_zero_statement(
        self, num_samples: int, deferred_calls: SeqCGenerator | None = None
    ) -> None: ...
    def add_play_hold_statement(
        self, num_samples: int, deferred_calls: SeqCGenerator | None = None
    ) -> None: ...
    def generate_seq_c(self) -> str: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...
    def __repr__(self) -> str: ...
    def compressed(self) -> SeqCGenerator: ...

def merge_generators(
    generators: Sequence[SeqCGenerator], compress: bool = True
) -> SeqCGenerator: ...

class SeqCTracker:
    def __init__(
        self,
        init_generator: SeqCGenerator,
        deferred_function_calls: SeqCGenerator,
        sampling_rate: float,
        delay: float,
        device_type: DeviceTypePy,
        signal_type: AWGSignalType,
        emit_timing_comments: bool,
        automute_playzeros_min_duration: float,
        automute_playzeros: bool = False,
    ) -> None: ...
    @property
    def automute_playzeros(self) -> bool: ...
    def add_required_playzeros(self, start: int) -> int: ...
    def flush_deferred_function_calls(self) -> None: ...
    def force_deferred_function_calls(self) -> None: ...
    def flush_deferred_phase_changes(self) -> None: ...
    def discard_deferred_phase_changes(self) -> None: ...
    def has_deferred_phase_changes(self) -> bool: ...
    def add_timing_comment(self, end_samples: int) -> None: ...
    def add_comment(self, comment: str) -> None: ...
    def add_function_call_statement(
        self,
        name: str,
        args: list[str] | None = None,
        assign_to: str | None = None,
        deferred: bool = False,
    ) -> None: ...
    def add_play_zero_statement(
        self, num_samples: int, increment_counter: bool = False
    ) -> None: ...
    def add_play_hold_statement(self, num_samples: int) -> None: ...
    def add_play_wave_statement(self, wave_id: str, channel: int | None) -> None: ...
    def add_command_table_execution(
        self,
        ct_index: int | str,
        latency: int | str | None = None,
        comment: str | None = "",
    ) -> None: ...
    def add_phase_change(self, ct_index: int, comment: str = "") -> None: ...
    def add_variable_assignment(
        self, variable_name: str, value: bool | int | float | str
    ) -> None: ...
    def add_variable_increment(
        self, variable_name: str, value: int | float
    ) -> None: ...
    def append_loop_stack_generator(
        self, always: bool = False, generator: SeqCGenerator | None = None
    ) -> SeqCGenerator: ...
    def push_loop_stack_generator(
        self, generator: SeqCGenerator | None = None
    ) -> None: ...
    def pop_loop_stack_generators(self) -> list[SeqCGenerator] | None: ...
    def setup_prng(
        self,
        seed: int | None = None,
        prng_range: int | None = None,
    ) -> None: ...
    def drop_prng(self) -> None: ...
    def add_prng_match_command_table_execution(self, offset: int) -> None: ...
    def sample_prng(self, declarations_generator: SeqCGenerator) -> None: ...
    def prng_tracker(self) -> PRNGTracker | None: ...
    def add_set_trigger_statement(self, value: int, deferred: bool = True) -> None: ...
    def add_startqa_shfqa_statement(
        self,
        generator_mask: str,
        integrator_mask: str,
        monitor: int | None = None,
        feedback_register: int | None = None,
        trigger: int | None = None,
    ) -> None: ...
    def trigger_output_state(self) -> int: ...
    @property
    def current_time(self) -> int: ...
    @current_time.setter
    def current_time(self, value: int) -> None: ...
    def top_loop_stack_generators_have_statements(self) -> bool: ...
    def commit_prng(self) -> None: ...

class PRNGTracker:
    def __init__(self) -> None: ...
    @property
    def offset(self) -> int: ...
    @offset.setter
    def offset(self, value: int) -> None: ...
    @property
    def active_sample(self) -> str | None: ...
    @active_sample.setter
    def active_sample(self, value: str) -> None: ...
    def drop_sample(self) -> None: ...
    def is_committed(self) -> bool: ...

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

class AwgCodeGenerationResult:
    awg_events: list[object]
    integration_weights: list[IntegrationWeight]
    has_readout_feedback: bool
    ppc_device: str | None
    ppc_channel: int | None
    global_delay: int = 0
    # Signal delays in seconds to be applied to the signal
    signal_delays: dict[str, float] = {}
    # Signal integration lengths in seconds
    # This is a mapping from signal name to SignalIntegrationInfo
    integration_lengths: dict[str, SignalIntegrationInfo] = {}
    feedback_register: int | Literal["local"] | None = None
    source_feedback_register: int | Literal["local"] | None = None

class SeqCGenOutput:
    """Output of the SeqC code generation process.

    Attributes:
        awg_results: List of code generation results for each awg.
        total_execution_time: Total execution time in seconds of the generated code.
        qa_signal_by_handle: Mapping from QA signal handles to their corresponding signal IDs and AWG keys.
        simultaneous_acquires: List of simultaneous acquisitions.
            Each element is a mapping from signal name to its acquisition handle which
            happen at the same time.
    """

    awg_results: list[AwgCodeGenerationResult]
    total_execution_time: float = 0.0
    qa_signal_by_handle: dict[str, tuple[str, AwgKey]] = {}
    simultaneous_acquires: list[dict[str, str]] = []

def generate_code(
    ir: IRTree,
    awgs: list[AWGInfo],
    settings: dict[str, object],
    waveform_sampler: object,
) -> SeqCGenOutput:
    """Generate SeqC code for given AWGs.

    Arguments:
        ir: The IR tree containing the data to be processed.
        awgs: List of target awgs.
        settings: Compiler settings as dictionary.
        waveform_sampler: Python object for waveform sampling.
    """
