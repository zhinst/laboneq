# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from numpy.typing import ArrayLike

from laboneq.compiler.scheduler.root_schedule import RootSchedule
from laboneq.core.types.enums.port_mode import PortMode
from laboneq.data.experiment_description import Experiment

class ExperimentInfo:
    """Object containing the information about the experiment."""

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

class Signal:
    def __init__(
        self,
        uid: str,
        sampling_rate: float,
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

def build_experiment(experiment: Experiment, signals: list[Signal]) -> ExperimentInfo:
    """Build a scheduled experiment.

    Args:
        experiment: Experiment description.
        signals: List of signals.

    Returns:
        An object containing the Rust experiment
    """

class ScheduledExperiment:
    """Result of an experiment scheduling.

    Attributes:
        used_parameters: Used near-time parameters in the experiment.
        root: Root scheduled node.
    """

    used_parameters: set[str]
    root: RootSchedule

def schedule_experiment(
    experiment: ExperimentInfo,
    parameters: dict[str, float],
    chunking_info: tuple[int, int] | None,
) -> ScheduledExperiment:
    """Schedule an experiment.

    Args:
        experiment: Experiment information.
        parameters: Dictionary of parameter values to be resolved.
        chunking_info: Tuple of (current chunk index, total chunk count) or None.

    Returns:
        Scheduled experiment.
    """
