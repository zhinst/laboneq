# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from numpy.typing import ArrayLike

from laboneq.compiler.scheduler.acquire_group_schedule import AcquireGroupSchedule
from laboneq.compiler.scheduler.loop_schedule import LoopSchedule
from laboneq.compiler.scheduler.match_schedule import MatchSchedule
from laboneq.compiler.scheduler.oscillator_schedule import (
    InitialLocalOscillatorFrequencySchedule,
    InitialOscillatorFrequencySchedule,
    OscillatorFrequencyStepSchedule,
)
from laboneq.compiler.scheduler.phase_reset_schedule import PhaseResetSchedule
from laboneq.compiler.scheduler.ppc_step_schedule import PPCStepSchedule
from laboneq.compiler.scheduler.pulse_schedule import (
    PrecompClearSchedule,
    PulseSchedule,
)
from laboneq.compiler.scheduler.section_schedule import SectionSchedule
from laboneq.compiler.scheduler.voltage_offset import InitialOffsetVoltageSchedule
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

class Schedules:
    """A compatibility layer for schedules between Python and Rust."""

    initial_oscillator_frequency: list[InitialOscillatorFrequencySchedule]
    initial_local_oscillator_frequency: list[InitialLocalOscillatorFrequencySchedule]
    initial_voltage_offset: list[InitialOffsetVoltageSchedule]
    # Set oscillator frequency steps per loop UID, where each list item corresponds to a local iteration number
    oscillator_frequency_steps: dict[str, list[OscillatorFrequencyStepSchedule]]
    # Phase reset scheduler per section UID, where each list item corresponds to a local iteration number (if loop)
    phase_resets: dict[str, list[PhaseResetSchedule]]
    # PPC step schedules per loop UID, where outermost list item corresponds to global iteration number,
    # and the next list item corresponds to local iteration number and innermost list to the PPC steps.
    ppc_steps: dict[str, list[list[list[PPCStepSchedule]]]]
    # Section, acquires in the order of depth-first traversal of the experiment tree
    acquire_schedules: dict[str, list[PulseSchedule | AcquireGroupSchedule]]
    # Section delays and precomp clears in the order of depth-first traversal of the experiment tree
    section_delays: dict[str, list[PulseSchedule | PrecompClearSchedule]]
    # Sections in the order of depth-first traversal of the experiment tree
    sections: dict[str, list[SectionSchedule]]
    # Section play pulse schedules in the order of depth-first traversal of the experiment tree
    play_pulse_schedules: dict[str, list[PulseSchedule]]
    loop_schedules: dict[str, list[LoopSchedule]]
    match_schedules: dict[str, list[MatchSchedule]]

class ScheduledExperiment:
    """Result of an experiment scheduling.

    Attributes:
        system_grid: System grid in tinysamples.
        used_parameters: Used near-time parameters in the experiment.
        schedules: Schedules object containing various schedules.
    """

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
