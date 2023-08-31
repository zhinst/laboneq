# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from laboneq.core.validators import dicts_equal
from laboneq.data.scheduled_experiment import ScheduledExperiment

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.data.recipe import Recipe
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.experiment.pulse import Pulse


@dataclass(init=True, repr=True, order=True)
class CompiledExperiment:
    """Data structure to store the output of the compiler.

    Attributes:
        device_setup (DeviceSetup):
            The device setup the experiment was compiled for.
        experiment (Experiment):
            The (uncompiled) experiment.
        experiment_dict (deprecated):
            Deprecated. A representation of the source experiment, using
            primitive Python datatypes only (dicts, lists, etc).
            Use `.experiment` instead.
        scheduled_experiment (internal):
            Internal. The internal representation of the compiled
            experiment. Available for debugging but subject to
            change in any LabOne Q release.

    !!! version-changed "Deprecated in version 2.14.0"
        The `.experiment_dict` attribute was deprecated in
        version 2.14.0. Use `.experiment` instead.

    !!! version-changed "Changed in version 2.14.0"
        The `.scheduled_experiment` attribute was documented to
        be internal and subject to change.
    """

    # The source device setup.
    device_setup: DeviceSetup | None = None

    # The source experiment.
    experiment: Experiment | None = None

    # A representation of the source experiment, using primitive Python datatypes only
    # (dicts, lists, etc.)
    experiment_dict: dict[str, Any] | None = None

    # Compiled
    scheduled_experiment: ScheduledExperiment | None = None

    # Proxy props for backwards compatibility
    @property
    def src(self) -> list[dict]:
        """The list of generated Sequencer C sources with one element
        for each sequencer core.

        !!! version-changed "Deprecated in version 2.14.0"
            For debugging, use `.scheduled_experiment.src` instead.

        Each element of the list is a dictionary with the keys:

        - `filename` ([str][]):
            The filename of the generated source.
        - `text` ([str][]):
            The generated source.
        """
        return self.scheduled_experiment.src

    @property
    def waves(self) -> list[dict]:
        """The list of sampled pulses generated for use within the
        Sequencer C programs.

        !!! version-changed "Deprecated in version 2.14.0"
            For debugging, use `.scheduled_experiment.waves` instead.

        Each element of the list is a dictionary with the keys:

        - `filename` ([str][]):
          The filename of the generated sample data.
        - `samples` ([numpy.ndarray][]):
          A one dimensional numpy array containing the sample
          data which may be either [float][] or [complex][].
        """
        return self.scheduled_experiment.waves

    @property
    def recipe(self) -> Recipe:
        """A part of the internal representation of the compiled
        experiment.

        !!! version-changed "Deprecated in version 2.14.0"
            For debugging, use `.scheduled_experiment.recipe` instead.
        """
        return self.scheduled_experiment.recipe

    @property
    def wave_indices(self) -> list[dict]:
        """A list of which waves are used by each Sequence C program.

        !!! version-changed "Deprecated in version 2.14.0"
            For debugging, use `.scheduled_experiment.wave_indices` instead.

        Each element of the list is a dictionary with the keys:

        - `filename` ([str][]):
          The filename of one of the Sequence C programs.
        - `value` ([dict][]):
          A mapping from the names of waves to pairs of `(channel, type)`
          where `channel` is an integer specifying the channel to play
          the wave samples on and `type` is either `"float"` or `"complex"`
          and specifies the type of the samples.
        """
        return self.scheduled_experiment.wave_indices

    @property
    def command_tables(self) -> list[dict]:
        """A list of command tables used by Sequence C programs.

        !!! version-changed "Deprecated in version 2.14.0"
            For debugging, use `.scheduled_experiment.command_tables`
            instead.

        Command table entries define custom real-time operations
        for the compiled program. The details of command table
        entries are an implementation detail that may change.

        Each element of the list is a dictionary with the keys:

        - `seqc` ([str][]):
          The filename of one of the Sequence C programs.
        - `ct` ([list][]):
          A list of command table entries. Each entry
          is a dictionary. The contents of the dictionary
          are an internal implementation detail.
        """
        return self.scheduled_experiment.command_tables

    @property
    def schedule(self) -> None:
        """Deprecated. Previously returned the internal scheduling
        information used by the compiler to generate
        Sequencer C for the devices.

        !!! version-changed "Deprecated in version 2.14.0"
            The `schedule` property returns [None][] since
            version 2.14.0. For debugging, use
            `.scheduled_experiment` instead.
        """
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
            pulse_uid:
                Pulse to replace, can be [Pulse][laboneq.dsl.experiment.pulse.Pulse]
                object or uid of the pulse
            pulse_or_array:
                Replacement pulse. May be a
                [Pulse][laboneq.dsl.experiment.pulse.Pulse] object or
                an array of values (see `sampled_pulse_*` from
                [pulse_library][laboneq.dsl.experiment.pulse_library])
        """
        from laboneq.core.utilities.replace_pulse import replace_pulse

        replace_pulse(self, pulse_uid, pulse_or_array)

    @classmethod
    def load(cls, filename: str) -> CompiledExperiment:
        """Load a compiled experiment from a JSON file.

        Args:
            filename: The file to load the compiled experiment from.
        """
        from laboneq.dsl.serialization import Serializer

        return Serializer.from_json_file(filename, cls)

    def save(self, filename: str):
        """Store a compiled experiment in a JSON file.

        Args:
            filename: The file to save the compiled experiment to.
        """
        from laboneq.dsl.serialization import Serializer

        Serializer.to_json_file(self, filename)
