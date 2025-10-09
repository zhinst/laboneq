# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.data.experiment_description import Experiment
from laboneq.compiler.common.awg_info import AwgKey

class ExperimentInfo:
    """Object containing the information about the experiment.

    This object cannot be send to threads and must be used in the same thread.
    """

class Signal:
    def __init__(self, uid: str, sampling_rate: float, device_uid: str, awg_index: int):
        """A representation of signal properties."""

def build_experiment(experiment: Experiment, signals: list[Signal]) -> ExperimentInfo:
    """Build a scheduled experiment.

    Args:
        experiment: Experiment description.
        signals: List of signals.

    Returns:
        An object containing the Rust experiment
    """

class ScheduledExperiment:
    """Result of an experiment scheduling."""

    max_acquisition_time_per_awg: dict[AwgKey, float]

def schedule_experiment(
    experiment: ExperimentInfo,
) -> ScheduledExperiment:
    """Schedule an experiment.

    Args:
        experiment: Experiment information.

    Returns:
        Scheduled experiment.
    """
