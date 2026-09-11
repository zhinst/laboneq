# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, Protocol

from laboneq.compiler.common.iface_compiler_output import RTCompilerOutput
from laboneq.core.types.units import Quantity
from laboneq.dsl.experiment import Section
from laboneq.dsl.parameter import Parameter

def init_logging(level: int) -> None:
    """Initialize logging with the given Python log level."""

class ProcessedExperiment:
    """Object containing the information about the experiment."""

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

class InternalConnection(Protocol):
    from_instrument: str
    from_port: str
    to_instrument: str
    to_port: str

class SetupDescriptionQccs(Protocol):
    instruments: list[Instrument]
    signals: list[DeviceSignal]
    internal_connections: list[InternalConnection]

class Experiment(Protocol):
    uid: str | None
    sections: list[Section]
    experiment_signals: list[ExperimentSignal]

def serialize_experiment(
    experiment: Experiment,
    setup_description: SetupDescriptionQccs,
    packed: bool = False,
) -> bytes:
    """Serialize an experiment to Cap'n Proto bytes.

    `setup_description` is stored as an opaque `DeviceSetup.setupDescription` pointer, in the
    QCCS setup-description schema. Other backends have their own compiler module.
    """

def compile_experiment(
    capnp_data: bytes,
    packed: bool = False,
    compiler_settings: dict | None = None,
) -> bytes:
    """Compile Cap'n Proto experiment bytes into `CompiledExperiment` Cap'n Proto bytes.

    `packed` describes `capnp_data` only; the returned bytes are always unpacked.
    """

class PulseSheetSchedule:
    """Opaque handle to a pulse sheet schedule computed by the real-time compiler.

    Carried opaquely through the near-time executor and only unpacked for the final compiler output.
    """

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
