# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.dsl.experiment import Experiment
from laboneq.dsl.parameter import Parameter

def init_logging(level: int) -> None:
    """Initialize logging with the given Python log level."""

class ExperimentInfo:
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

class DeviceSetupBuilder:
    def __init__(self): ...
    def add_instrument(
        self,
        uid: str,
        device_uid: str,
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

class OscillatorRef:
    """Reference to a oscillator."""

class OutputRoute:
    """A representation of an output route."""

class AmplifierPump:
    def __init__(
        self,
        device: str,
        channel: int,
        pump_frequency: float | Parameter | None = None,
        pump_power: float | Parameter | None = None,
        cancellation_phase: float | Parameter | None = None,
        cancellation_attenuation: float | Parameter | None = None,
        probe_frequency: float | Parameter | None = None,
        probe_power: float | Parameter | None = None,
    ):
        """A representation of an amplifier pump configuration."""

class Precompensation:
    def __init__(
        self,
        exponential: list[ExponentialCompensation] = [],
        high_pass: HighPassCompensation | None = None,
        bounce: BounceCompensation | None = None,
        fir: FirCompensation | None = None,
    ): ...

class ExponentialCompensation:
    def __init__(timeconstant: float, amplitude: float): ...

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
    experiment: Experiment,
    device_setup: DeviceSetupBuilder,
    packed: bool = False,
) -> bytes:
    """Serialize an experiment to Cap'n Proto bytes."""

def build_experiment_capnp(
    capnp_data: bytes,
    awgs: list[AwgInfo],
    desktop_setup: bool,
    packed: bool = False,
) -> ExperimentInfo:
    """Build a scheduled experiment from Cap'n Proto bytes."""

class ExperimentIr:
    """A representation of the experiment IR."""

class ScheduleResult:
    """Result of an experiment scheduling.

    Attributes:
        used_parameters: Used near-time parameters in the experiment.
        experiment_ir: Experiment IR.
    """

    experiment_ir: ExperimentIr
    used_parameters: set[str]

def schedule_experiment(
    experiment: ExperimentInfo,
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

def generate_pulse_sheet_schedule(
    ir_py: ExperimentIr,
    expand_loops: bool,
    max_events: int,
) -> dict:
    """Generate a schedule (event list + metadata) from an IR tree.

    This function is used by the Python compiler to generate the event list
    for the Pulse Sheet Viewer (PSV).

    Args:
        ir_py: The experiment IR from Python.
        expand_loops: Whether to expand compressed loops (EXPAND_LOOPS_FOR_SCHEDULE flag).
        max_events: Maximum number of events to generate (MAX_EVENTS_TO_PUBLISH setting).

    Returns:
        A Python dict containing:
        - event_list: List of scheduler events
        - event_list_truncated: Whether event generation hit the MAX_EVENTS_TO_PUBLISH limit
        - section_info: Section metadata with preorder map
        - section_signals_with_children: Signal hierarchy per section
        - sampling_rates: Sampling rates per device type
    """
