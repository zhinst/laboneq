# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, Protocol, TypedDict

from laboneq.compiler.common.iface_compiler_output import (
    CombinedOutput,
    RTCompilerOutput,
)
from laboneq.core.types.numpy_support import NumPyArray
from laboneq.core.types.units import Quantity
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.dsl.experiment import Section
from laboneq.dsl.parameter import Parameter

def init_logging(level: int) -> None:
    """Initialize logging with the given Python log level."""

class ProcessedExperiment:
    """Object containing the information about the experiment."""

    def device_lead_delay(self, device_uid: str) -> float: ...
    def signal_delay_compensation(self, signal_uid: str) -> float: ...
    def signal_precompensation(self, signal_uid: str) -> dict[str, dict] | None: ...
    def get_result_shapes(
        self, combined_output: CombinedOutput
    ) -> list[HandleResultShape]: ...
    def rt_loop_properties(self) -> RtLoopProperties: ...

class RtLoopProperties:
    """Properties of the real-time loop."""

    uid: str
    acquisition_type: Literal[
        "INTEGRATION",
        "RAW",
        "DISCRIMINATION",
        "SPECTROSCOPY_IQ",
        "SPECTROSCOPY_PSD",
        "SPECTROSCOPY",
    ]
    averaging_mode: Literal["CYCLIC", "SEQUENTIAL", "SINGLE_SHOT"]
    count: int

class HandleResultShape:
    handle: str
    shape: list[int]
    axis_names: list[list[str]]
    axis_values: list[list[NumPyArray]]
    chunked_axis_index: int | None
    match_case_mask: dict[int, list[int]]

# ---------------------------------------------------------------------------
# Input DTOs for serialize_experiment()
# All types are structurally typed (duck-typed via FromPyObject in Rust).
# SimpleNamespace or any object with matching attributes is accepted.
# ---------------------------------------------------------------------------

class Oscillator(Protocol):
    uid: str
    frequency: float | Parameter
    modulation: Literal["AUTO", "HARDWARE", "SOFTWARE"] | None

class AmplifierPump(Protocol):
    device: str
    channel: int
    alc_on: bool
    pump_on: bool
    pump_filter_on: bool
    pump_power: float | Parameter | None
    pump_frequency: float | Parameter | None
    probe_on: bool
    probe_power: float | Parameter | None
    probe_frequency: float | Parameter | None
    cancellation_on: bool
    cancellation_phase: float | Parameter | None
    cancellation_attenuation: float | Parameter | None
    cancellation_source: CancellationSource
    cancellation_source_frequency: float | None

class CancellationSource(Enum):
    INTERNAL = 0
    EXTERNAL = 1

class OutputRoute(Protocol):
    source_signal: str
    amplitude_scaling: float | Parameter | None
    phase_shift: float | Parameter | None

class HighPassCompensation(Protocol):
    timeconstant: float

class ExponentialCompensation(Protocol):
    timeconstant: float
    amplitude: float

class FirCompensation(Protocol):
    coefficients: list[float]
    strict: bool

class BounceCompensation(Protocol):
    delay: float
    amplitude: float

class Precompensation(Protocol):
    high_pass: HighPassCompensation | None
    exponential: list[ExponentialCompensation] | None
    fir: FirCompensation | None
    bounce: BounceCompensation | None

class MixerCalibration(Protocol):
    voltage_offsets: list[float | Parameter] | None
    correction_matrix: list[list[float | Parameter]] | None

class ExperimentSignal(Protocol):
    # Experiment signal UID
    uid: str
    # The device signal UID this experiment signal maps to
    maps_to: str

    # Calibration
    amplitude: float | Parameter | None
    oscillator: Oscillator | None
    lo_frequency: float | Parameter | None
    voltage_offset: float | Parameter | None
    amplifier_pump: AmplifierPump | None
    port_mode: Literal["RF", "LF"] | None
    automute: bool
    delay_signal: float
    port_delay: float | Parameter | None
    range: Quantity | None
    precompensation: Precompensation | None
    added_outputs: list[OutputRoute]
    threshold: list[float] | None
    mixer_calibration: MixerCalibration | None

class Instrument(Protocol):
    uid: str
    device_type: str
    options: list[str]
    reference_clock_source: str | None

class DeviceSignal(Protocol):
    uid: str
    ports: list[str]
    instrument_uid: str

class SetupDescriptionQccs(Protocol):
    instruments: list[Instrument]
    signals: list[DeviceSignal]

class ChannelConfig(Protocol):
    geolocation: str
    channel_type: ChannelType

class ChannelType(Enum):
    RF = 0
    QA = 1
    FLUX = 2

class SetupDescriptionZqcs(Protocol):
    uid: str
    data: bytes
    channels: list[ChannelConfig]

class Experiment(Protocol):
    uid: str | None
    sections: list[Section]
    experiment_signals: list[ExperimentSignal]

class DeviceSetup(Protocol):
    setup_description: SetupDescriptionQccs | SetupDescriptionZqcs

def serialize_experiment(
    experiment: Experiment,
    device_setup: DeviceSetup,
    packed: bool = False,
) -> bytes:
    """Serialize an experiment to Cap'n Proto bytes."""

def compile_experiment(
    capnp_data: bytes,
    packed: bool = False,
    compiler_settings: dict | None = None,
) -> ScheduledExperiment:
    """Build a scheduled experiment from Cap'n Proto bytes."""

class PulseSheetSchedule(TypedDict):
    """A representation of the pulse sheet schedule for the Pulse Sheet Viewer.

    Attributes:
        event_list: List of scheduler events.
        event_list_truncated: Whether event generation hit the MAX_EVENTS_TO_PUBLISH limit.
        section_info: Section metadata with preorder map.
        section_signals_with_children: Signal hierarchy per section.
        sampling_rates: Sampling rates per device type.
    """

    event_list: list[dict]
    event_list_truncated: bool
    section_info: dict
    section_signals_with_children: dict
    sampling_rates: dict

class RealTimeCompilerOutput:
    """Result of a real-time compilation.

    Attributes:
        code_gen_output: Code generation output.
        used_parameters: Used near-time parameters in the experiment.
        pulse_sheet_schedule: Optional pulse sheet schedule (event list + metadata) for the Pulse Sheet Viewer.
    """

    code_gen_output: RTCompilerOutput
    used_parameters: set[str]
    pulse_sheet_schedule: PulseSheetSchedule | None

def compile_realtime(
    experiment: ProcessedExperiment,
    parameters: dict[str, float],
    chunking_info: tuple[int, int] | None,
) -> RealTimeCompilerOutput:
    """Compile real-time experiment.

    Args:
        experiment: Experiment information.
        parameters: Dictionary of parameter values to be resolved.
        chunking_info: Tuple of (current chunk index, total chunk count) or None.

    Returns:
        Compiled real-time experiment.
    """

class SpanBuffer:
    """A buffer for collecting spans from LabOne Q Rust components."""

    def flush_spans(self) -> list[str]:
        """Flush the collected spans as a list of JSON strings."""
