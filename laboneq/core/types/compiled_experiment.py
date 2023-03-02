# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from laboneq.core.types.enums.mixer_type import MixerType
from laboneq.core.validators import dicts_equal

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.experiment import Experiment
    from laboneq.dsl.experiment.pulse import Pulse


@dataclass
class PulseInstance:
    offset_samples: int
    amplitude: float = None  # instance (final) amplitude
    length: float = None  # instance (final) length
    iq_phase: float = None
    modulation_frequency: float = None
    modulation_phase: float = None
    channel: int = None  # The AWG channel for rf_signals
    needs_conjugate: bool = False  # SHF devices need that for now
    play_pulse_parameters: Optional[Dict[str, Any]] = field(default_factory=dict)
    pulse_pulse_parameters: Optional[Dict[str, Any]] = field(default_factory=dict)

    # uid of pulses that this instance overlaps with
    overlaps: List[str] = None
    has_marker1: bool = False
    has_marker2: bool = False


@dataclass
class PulseWaveformMap:
    """Data structure to store mappings between the given pulse and an AWG waveform."""

    sampling_rate: float
    length_samples: int
    signal_type: str
    # UHFQA's HW modulation is not an IQ mixer. None for flux pulses etc.
    mixer_type: Optional[MixerType] = field(default=None)
    instances: List[PulseInstance] = field(default_factory=list)


@dataclass
class PulseMapEntry:
    """Data structure to store the :py:class:`PulseWaveformMap` of each AWG waveform."""

    # key: waveform signature string
    #: A mapping of signals to :py:class:`PulseWaveformMap`
    waveforms: Dict[str, PulseWaveformMap] = field(default_factory=dict)


@dataclass(init=True, repr=True, order=True)
class CompiledExperiment:
    """Data structure to store the output of the compiler."""

    #: The source device setup.
    device_setup: DeviceSetup = field(default=None)

    #: The source experiment.
    experiment: Experiment = field(default=None)

    #: Instructions to the controller for running the experiment.
    recipe: Dict[str, Any] = field(default=None)

    #: The seqC source code, per device.
    src: List[Dict[str, str]] = field(default=None)

    #: The waveforms that will be uploaded to the devices.
    waves: List[Dict[str, Any]] = field(default=None)

    #: Data structure for storing the indices or filenames by which the waveforms are
    #: referred to during and after upload.
    wave_indices: List[Dict[str, Any]] = field(default=None)

    #: Datastructure for storing the command table data
    command_tables: List[Dict[str, Any]] = field(default_factory=list)

    #: List of events as scheduled by the compiler.
    schedule: Dict[str, Any] = field(default=None)

    #: A representation of the source experiment, using primitive Python datatypes only
    #: (dicts, lists, etc.)
    experiment_dict: Dict[str, Any] = field(default=None)

    #: Data structure for mapping pulses (in the experiment) to waveforms (on the
    #: device).
    pulse_map: Dict[str, PulseMapEntry] = field(default=None)

    def __eq__(self, other: CompiledExperiment):
        if self is other:
            return True

        if len(self.waves) != len(other.waves):
            return False

        return (
            self.experiment == other.experiment
            and self.recipe == other.recipe
            and self.src == other.src
            and self.wave_indices == other.wave_indices
            and self.schedule == other.schedule
            and dicts_equal(other.experiment_dict, self.experiment_dict)
            and dicts_equal(other.waves, self.waves)
            and self.pulse_map == other.pulse_map
        )

    def replace_pulse(
        self, pulse_uid: Union[str, Pulse], pulse_or_array: "Union[ArrayLike, Pulse]"
    ):
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
