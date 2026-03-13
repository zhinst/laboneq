# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from numpy.typing import ArrayLike

from laboneq.core.types.enums.port_mode import PortMode
from laboneq.dsl.experiment import Experiment

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

class SweepParameter:
    def __init__(self, uid: str, values: ArrayLike, driven_by: list[str]):
        """A representation of a sweep parameter."""

class Oscillator:
    def __init__(self, uid: str, frequency: float | SweepParameter, is_hardware: bool):
        """A representation of an oscillator."""

class AmplifierPump:
    def __init__(
        self,
        device: str,
        channel: int,
        pump_frequency: float | SweepParameter | None = None,
        pump_power: float | SweepParameter | None = None,
        cancellation_phase: float | SweepParameter | None = None,
        cancellation_attenuation: float | SweepParameter | None = None,
        probe_frequency: float | SweepParameter | None = None,
        probe_power: float | SweepParameter | None = None,
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

class OutputRoute:
    def __init__(
        self,
        source_channel: int,
        amplitude_scaling: float | SweepParameter,
        phase_shift: float | SweepParameter,
    ): ...

class Signal:
    def __init__(
        self,
        uid: str,
        awg_key: int,
        device_uid: str,
        oscillator: Oscillator | None,
        lo_frequency: float | SweepParameter | None,
        voltage_offset: float | SweepParameter | None,
        kind: Literal["RF", "IQ", "INTEGRATION"],
        amplifier_pump: AmplifierPump | None,
        channels: list[int],
        port_mode: PortMode | None,
        automute: bool,
        signal_delay: float,
        port_delay: float | SweepParameter,
        # value, unit (e.g. (1.0, 'volt')) or None
        range: tuple[float, str] | None,
        precompensation: Precompensation | None,
        added_outputs: list[OutputRoute],
    ):
        """A representation of signal properties."""

class Device:
    def __init__(
        self,
        uid: str,
        physical_device_uid: int,
        kind: str,
        is_shfqc: bool,
    ):
        """A representation of device properties."""

class AwgInfo:
    def __init__(self, uid: int, number: list[int]):
        """A representation of AWG properties."""

def build_experiment(
    experiment: Experiment,
    signals: list[Signal],
    devices: list[Device],
    awgs: list[AwgInfo],
    desktop_setup: bool,
) -> ExperimentInfo:
    """Build a scheduled experiment.

    Args:
        experiment: Experiment description.
        signals: List of signals.
        devices: List of devices.
        awgs: List of AWG information.
        desktop_setup: Whether the experiment is being built for a desktop setup.

    Returns:
        An object containing the Rust experiment
    """

def serialize_experiment(
    experiment: Experiment,
    packed: bool = False,
) -> bytes:
    """Serialize an experiment to Cap'n Proto bytes."""

def build_experiment_capnp(
    capnp_data: bytes,
    signals: list[Signal],
    devices: list[Device],
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
        - section_info: Section metadata with preorder map
        - section_signals_with_children: Signal hierarchy per section
        - sampling_rates: Sampling rates per device type
    """
