# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from laboneq.core.validators import dicts_equal
from laboneq.data.scheduled_experiment import ScheduledExperiment

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.experiment.pulse import Pulse


@dataclass(init=True, repr=True, order=True)
class CompiledExperiment:
    """Data structure to store the output of the compiler."""

    #: The source device setup.
    device_setup: DeviceSetup | None = None

    #: The source experiment.
    experiment: Experiment | None = None

    #: A representation of the source experiment, using primitive Python datatypes only
    #: (dicts, lists, etc.)
    experiment_dict: dict[str, Any] | None = None

    #: Compiled
    scheduled_experiment: ScheduledExperiment | None = None

    # Proxy props for backwards compatibility
    @property
    def src(self):
        return self.scheduled_experiment.src

    @property
    def waves(self):
        return self.scheduled_experiment.waves

    @property
    def recipe(self):
        return self.scheduled_experiment.recipe

    @property
    def wave_indices(self):
        return self.scheduled_experiment.wave_indices

    @property
    def command_tables(self):
        return self.scheduled_experiment.command_tables

    @property
    def schedule(self):
        return self.scheduled_experiment.schedule

    def __eq__(self, other):
        if other is self:
            return True
        if type(other) is not CompiledExperiment:
            return NotImplemented
        return (other.device_setup, other.experiment, other.scheduled_experiment) == (
            self.device_setup,
            self.experiment,
            self.scheduled_experiment,
        ) and dicts_equal(other.experiment_dict, self.experiment_dict)

    def replace_pulse(self, pulse_uid: str | Pulse, pulse_or_array: ArrayLike | Pulse):
        """Permanently replaces specific pulse with the new sample data in the compiled
        experiment. Previous pulse data is lost.

        Args:
            pulse_uid: pulse to replace, can be :py:class:`~.dsl.experiment.pulse.Pulse`
                object or uid of the pulse
            pulse_or_array: replacement pulse, can be
                :py:class:`~.dsl.experiment.pulse.Pulse` object or value array (see
                ``sampled_pulse_*`` from the :py:mod:`~.dsl.experiment.pulse_library`)
        """
        from laboneq.core.utilities.replace_pulse import replace_pulse

        replace_pulse(self, pulse_uid, pulse_or_array)

    @classmethod
    def load(cls, filename) -> CompiledExperiment:
        """Load a compiled experiment from a JSON file."""
        from laboneq.dsl.serialization import Serializer

        return Serializer.from_json_file(filename, cls)

    def save(self, filename):
        """Store a compiled experiment in a JSON file."""
        from laboneq.dsl.serialization import Serializer

        Serializer.to_json_file(self, filename)
