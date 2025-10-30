# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import ArrayLike

from laboneq.data.experiment_description import Experiment
from laboneq.compiler.common.awg_info import AwgKey
from laboneq.compiler.scheduler.oscillator_schedule import (
    InitialOscillatorFrequencySchedule,
    InitialLocalOscillatorFrequencySchedule,
)
from laboneq.compiler.scheduler.voltage_offset import InitialOffsetVoltageSchedule

class ExperimentInfo:
    """Object containing the information about the experiment."""

class SweepParameter:
    def __init__(self, uid: str, values: ArrayLike, driven_by: list[str]):
        """A representation of a sweep parameter."""

class Oscillator:
    def __init__(self, uid: str, frequency: float | SweepParameter, is_hardware: bool):
        """A representation of an oscillator."""

class Signal:
    def __init__(
        self,
        uid: str,
        sampling_rate: float,
        awg_key: int,
        device: str,
        oscillator: Oscillator | None,
        lo_frequency: float | SweepParameter | None,
        voltage_offset: float | SweepParameter | None,
    ):
        """A representation of signal properties."""

def build_experiment(experiment: Experiment, signals: list[Signal]) -> ExperimentInfo:
    """Build a scheduled experiment.

    Args:
        experiment: Experiment description.
        signals: List of signals.

    Returns:
        An object containing the Rust experiment
    """

class Schedules:
    """A compatibility layer for schedules between Python and Rust."""

    initial_oscillator_frequency: list[InitialOscillatorFrequencySchedule]
    initial_local_oscillator_frequency: list[InitialLocalOscillatorFrequencySchedule]
    initial_voltage_offset: list[InitialOffsetVoltageSchedule]

class RepetitionInfo:
    mode: str  # "fastest", "constant", "auto"
    time: float | None
    loop_uid: str

class ScheduledExperiment:
    """Result of an experiment scheduling.

    Attributes:
        max_acquisition_time_per_awg: Maximum acquisition time per AWG in seconds.
        repetition_info: Repetition information of the experiment.
        system_grid: System grid in tinysamples.
        used_parameters: Used near-time parameters in the experiment.
        schedules: Schedules object containing various schedules.
    """

    max_acquisition_time_per_awg: dict[AwgKey, float]
    repetition_info: RepetitionInfo | None
    system_grid: int
    used_parameters: set[str]
    schedules: Schedules

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
