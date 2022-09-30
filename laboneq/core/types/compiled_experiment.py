# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Union, Optional, TYPE_CHECKING
import numpy as np

from laboneq.core.utilities.replace_pulse import replace_pulse

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.device_output_signals import DeviceOutputSignals

if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse
    from laboneq.dsl.experiment import Experiment
    from numpy.typing import ArrayLike


@dataclass
class PulseInstance:
    offset_samples: int
    amplitude: float = None  # instance (final) amplitude
    iq_phase: float = None
    modulation_frequency: float = None
    modulation_phase: float = None
    channel: int = None  # The AWG channel for rf_signals
    needs_conjugate: bool = False  # SHF devices need that for now

    # uid of pulses that this instance overlaps with
    overlaps: List[str] = None


@dataclass
class PulseWaveformMap:
    """Data structure to store mappings between the given pulse and an AWG waveform."""

    sampling_rate: float
    length_samples: int
    signal_type: str
    complex_modulation: bool  # UHFQA does not allow complex wave forms
    instances: List[PulseInstance] = field(default_factory=list)


@dataclass
class PulseMapEntry:
    """Data structure to store the :py:class:`PulseWaveformMap` of each AWG waveform.
    """

    # key: waveform signature string
    #: A mapping of signals to :py:class:`PulseWaveformMap`
    waveforms: Dict[str, PulseWaveformMap] = field(default_factory=dict)


@dataclass(init=True, repr=True, order=True)
class CompiledExperiment:
    """Data structure to store the output of the compiler.
    """

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

    #: List of events as scheduled by the compiler.
    schedule: Dict[str, Any] = field(default=None)

    #: A representation of the source experiment, using primitive Python datatypes only
    #: (dicts, lists, etc.)
    experiment_dict: Dict[str, Any] = field(default=None)

    #: Simulated output of the devices. Optional, only populated after calling
    #: :py:meth:`Session.compile() <.dsl.session.Session.compile>` or
    #: :py:meth:`Session.run() <.dsl.session.Session.run>` with
    #: ``do_simulation=True``, or by calling :py:meth:`simulate_outputs()`.
    output_signals: Optional[DeviceOutputSignals] = field(default=None)

    #: Data structure for mapping pulses (in the experiment) to waveforms (on the
    #: device).
    pulse_map: Dict[str, PulseMapEntry] = field(default=None)

    def __eq__(self, other: CompiledExperiment):
        if self is other:
            return True

        if len(self.waves) != len(other.waves):
            return False

        for i, wave in enumerate(self.waves):
            other_wave = other.waves[i]
            if other_wave.get("filename") != wave.get("filename"):
                return False
            if "samples" in wave:
                if not "samples" in other_wave:
                    return False
                if not np.allclose(wave["samples"], other_wave["samples"]):
                    return False
            elif "samples" in other_wave:
                return False

        return (
            self.experiment == other.experiment
            and self.recipe == other.recipe
            and self.src == other.src
            and self.wave_indices == other.wave_indices
            and self.schedule == other.schedule
            and self.experiment_dict == other.experiment_dict
            and self.output_signals == other.output_signals
            and self.pulse_map == other.pulse_map
        )

    def simulate_outputs(self, max_simulation_time=10e-6, log_level=None):
        """Simulate the output of the devices.

        This will populate the :py:attr:`output_signals` field.

        Args:
            max_simulation_time (float): The time after which the simulation is
                truncated.
            log_level (int): The log level when running the simulator. When `None`, the
                log level is left unchanged.


        See Also:
            - :py:meth:`Session.compile() <.dsl.session.Session.compile>`
            - :py:meth:`Session.run() <.dsl.session.Session.run>`
        """
        if self.output_signals is not None:
            return

        # delayed import because of circular references
        from laboneq.dsl.laboneq_facade import LabOneQFacade

        logger = logging.getLogger(__name__)
        if log_level is not None:
            logger.setLevel(log_level)

        self.output_signals = LabOneQFacade.simulate_outputs(
            self, max_simulation_time, logger
        )

    def get_output_signals(self, device_uid: str):
        res = [
            sig
            for sig in self.output_signals.signals
            if sig["device_uid"] == device_uid
        ]
        if not res:
            raise LabOneQException(f"No output for device: {device_uid}")
        return res

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
