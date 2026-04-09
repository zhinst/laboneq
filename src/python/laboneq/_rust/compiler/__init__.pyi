# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import TypedDict

from laboneq.dsl.experiment import Experiment as DslExperiment
from laboneq.dsl.parameter import Parameter

def init_logging(level: int) -> None:
    """Initialize logging with the given Python log level."""

class Experiment:
    """Object containing the information about the experiment."""

    def signals(self) -> list[str]: ...
    def signal_device_uid(self, signal_uid: str) -> str: ...
    def signal_sampling_rate(self, signal_uid: str) -> float: ...
    def device_lead_delay(self, physical_device_uid: int) -> float: ...
    def signal_delay_compensation(self, signal_uid: str) -> float: ...
    def signal_hw_oscillator(
        self, signal_uid: str
    ) -> tuple[str, float | None, str | None] | None:
        """Return the hardware oscillator information for a signal, if it exists.

        Returns:
            A tuple of (oscillator_id, fixed frequency, parameter uid) if a hardware oscillator is associated with the signal, or None otherwise.

                Either fixed frequency or parameter uid can be None, but not both.
        """
    def signal_precompensation(self, signal_uid: str) -> Precompensation | None: ...
    def signal_automute(self, signal_uid: str) -> bool: ...

class DeviceSetupBuilder:
    def __init__(self): ...
    def add_instrument(
        self,
        uid: str,
        device_type: str,
        physical_device_uid: int,
        options: list[str] | None = None,
        reference_clock_source: str | None = None,
        is_shfqc: bool = False,
    ) -> None: ...
    def add_signal_with_calibration(
        self,
        uid: str,
        ports: list[str],
        instrument_uid: str,
        channel_type: str,
        awg_core: int,
        amplitude: float | Parameter | None = None,
        oscillator: OscillatorRef | None = None,
        lo_frequency: float | Parameter | None = None,
        voltage_offset: float | Parameter | None = None,
        amplifier_pump: AmplifierPump | None = None,
        port_mode: str | None = None,
        automute: bool = False,
        signal_delay: float = 0.0,
        port_delay: float | Parameter | None = None,
        range: tuple[float, str | None] | None = None,
        precompensation: Precompensation | None = None,
        added_outputs: list[OutputRoute] | None = None,
        threshold: list[float] | None = None,
        mixer_calibration: MixerCalibrationRef | None = None,
    ) -> None: ...
    def create_oscillator(
        self,
        uid: str,
        frequency: float | Parameter,
        modulation: str | None = None,
    ) -> OscillatorRef: ...
    def create_output_route(
        self,
        source_signal: str,
        amplitude_scaling: float | Parameter | None = None,
        phase_shift: float | Parameter | None = None,
    ) -> OutputRoute: ...
    def create_mixer_calibration(
        self,
        voltage_offsets: list,
        correction_matrix: list[list],
    ) -> MixerCalibrationRef: ...

class OscillatorRef:
    """Reference to a oscillator."""

class MixerCalibrationRef:
    """Reference to a mixer calibration configuration."""

class OutputRoute:
    """A representation of an output route."""

class AmplifierPump:
    def __init__(
        self,
        device: str,
        channel: int,
        pump_power: float | Parameter | None,
        pump_frequency: float | Parameter | None,
        probe_power: float | Parameter | None,
        probe_frequency: float | Parameter | None,
        cancellation_phase: float | Parameter | None,
        cancellation_attenuation: float | Parameter | None,
    ):
        """A representation of an amplifier pump configuration."""

class Precompensation:
    def __init__(
        self,
        high_pass: HighPassCompensation | None = None,
        exponential: list[ExponentialCompensation] = [],
        fir: FirCompensation | None = None,
        bounce: BounceCompensation | None = None,
    ): ...

class ExponentialCompensation:
    def __init__(self, timeconstant: float, amplitude: float): ...

class HighPassCompensation:
    def __init__(self, timeconstant: float): ...

class FirCompensation:
    def __init__(self, coefficients: list[float]): ...

class BounceCompensation:
    def __init__(self, delay: float, amplitude: float): ...

class AwgInfo:
    def __init__(self, uid: int, number: list[int]):
        """A representation of AWG properties."""

def serialize_experiment(
    experiment: DslExperiment,
    device_setup: DeviceSetupBuilder,
    packed: bool = False,
) -> bytes:
    """Serialize an experiment to Cap'n Proto bytes."""

def build_experiment_capnp(
    capnp_data: bytes,
    awgs: list[AwgInfo],
    desktop_setup: bool,
    packed: bool = False,
) -> Experiment:
    """Build a scheduled experiment from Cap'n Proto bytes."""

class ExperimentIr:
    """A representation of the experiment IR."""

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

class ScheduleResult:
    """Result of an experiment scheduling.

    Attributes:
        used_parameters: Used near-time parameters in the experiment.
        experiment_ir: Experiment IR.
        pulse_sheet_schedule: Optional pulse sheet schedule (event list + metadata) for the Pulse Sheet Viewer.
    """

    experiment_ir: ExperimentIr
    used_parameters: set[str]
    pulse_sheet_schedule: PulseSheetSchedule | None

def schedule_experiment(
    experiment: Experiment,
    parameters: dict[str, float],
    chunking_info: tuple[int, int] | None,
) -> ScheduleResult:
    """Schedule an experiment.

    Args:
        experiment: Experiment information.
        parameters: Dictionary of parameter values to be resolved.
        chunking_info: Tuple of (current chunk index, total chunk count) or None.

    Returns:
        Scheduled experiment.
    """

class SpanBuffer:
    """A buffer for collecting spans from LabOne Q Rust components."""

    def flush_spans(self) -> list[str]:
        """Flush the collected spans as a list of JSON strings."""
