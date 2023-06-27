# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.dsl_dataclass_decorator import classformatter
from laboneq.dsl.experiment import ExperimentSignal
from laboneq.dsl.serialization import Serializer


class SignalType(Enum):
    DRIVE = "drive"
    MEASURE = "measure"
    ACQUIRE = "acquire"
    FLUX = "flux"


class QuantumElementSignalMap(MutableMapping):
    def __init__(
        self, items: Dict[str, str], key_validator: Callable[[str], None] = None
    ) -> None:
        """A mapping between signal.

        Args:
            items: Mapping between the signal names.
            key_validator: Callable to validate mapping keys.
        """
        self._items = {}
        self._key_validator = key_validator
        if self._key_validator:
            for k, v in items.items():
                self._items[self._key_validator(k)] = v
        else:
            self._items = items

    def __getitem__(self, key: Any):
        return self._items[key]

    def __setitem__(self, key: Any, value: Any):
        if self._key_validator:
            self._key_validator(key)
        self._items[key] = value

    def __delitem__(self, key: Any):
        del self._items[key]

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, dict):
            return __o == self._items
        return super().__eq__(__o)

    def __repr__(self):
        return repr(self._items)


@classformatter
@dataclass(init=False, repr=True)
class QuantumElement(ABC):
    """An abstract base class for quantum elements."""

    uid: str
    signals: Dict[str, str]
    parameters: Dict[str, Any]

    def __init__(
        self,
        uid: str = None,
        signals: Dict[str, LogicalSignal] = None,
        parameters: Dict[str, Any] = None,
    ):
        """
        Initializes a new QuantumElement object.

        Args:
            uid: A unique identifier for the quantum element.
            signals: A dictionary of logical signals associated with the quantum element.
            parameters: A dictionary of parameters associated with the quantum element.
        """
        self.uid = uuid.uuid4().hex if uid is None else uid
        if signals is None:
            signals = QuantumElementSignalMap({})
        if isinstance(signals, dict):
            sigs = {
                k: self._resolve_to_logical_signal_uid(v) for k, v in signals.items()
            }
            self.signals = QuantumElementSignalMap(sigs)
        else:
            self.signals = signals
        self._parameters = {} if parameters is None else parameters

    def __hash__(self):
        return hash(self.uid)

    @staticmethod
    def _resolve_to_logical_signal_uid(signal: Union[str, LogicalSignal]) -> str:
        return signal.path if isinstance(signal, LogicalSignal) else signal

    @property
    def parameters(self):
        """Parameters of the element."""
        return self._parameters

    @classmethod
    def load(cls, filename: Union[str, bytes, os.PathLike]) -> "QuantumElement":
        """
        Loads a QuantumElement object from a JSON file.

        Args:
            filename: The name of the JSON file to load the QuantumElement object from.
        """
        return cls.from_json(filename)

    @classmethod
    def from_json(cls, filename: Union[str, bytes, os.PathLike]) -> "QuantumElement":
        """Loads a QuantumElement object from a JSON file.

        Args:
            filename: The name of the JSON file to load the QuantumElement object from.
        """
        return Serializer.from_json_file(filename, cls)

    def save(self, filename: Union[str, bytes, os.PathLike]):
        """
        Save a QuantumElement object to a JSON file.

        Args:
            filename: The name of the JSON file to save the QuantumElement object.
        """
        self.to_json(filename)

    def to_json(self, filename: Union[str, bytes, os.PathLike]):
        """
        Save a QuantumElement object to a JSON file.

        Args:
            filename: The name of the JSON file to save the QuantumElement object.
        """
        Serializer.to_json_file(self, filename)

    def add_signals(self, signals: Dict[str, LogicalSignal]):
        """
        Adds logical signals to the quantum element.

        Args:
            signals: A dictionary of logical signals to add to the quantum element.
        """
        self.signals.update(
            {k: self._resolve_to_logical_signal_uid(v) for (k, v) in signals.items()}
        )

    @abstractmethod
    def calibration(self) -> Calibration:
        """Calibration of the Quantum element."""
        pass

    def experiment_signals(self) -> List[ExperimentSignal]:
        """Experiment signals of the quantum element."""
        sigs = []
        for k, v in self.calibration().items():
            sig = ExperimentSignal(
                uid=k,
                calibration=v,
                map_to=k,
            )
            sigs.append(sig)
        return sigs


@classformatter
@dataclass
class QubitParameters:
    #: Resonance frequency of the qubit.
    res_frequency: float
    #: Local oscillator frequency.
    lo_frequency: float
    #: Readout resonance frequency of the qubit.
    readout_res_frequency: float
    #: Readout local oscillator frequency.
    readout_lo_frequency: float
    #: Free form dictionary of user defined parameters.
    user_defs: Dict = field(default_factory=dict)

    @property
    def drive_frequency(self) -> float:
        """Qubit drive frequency."""
        return self.res_frequency - self.lo_frequency

    @property
    def readout_frequency(self) -> float:
        """Readout baseband frequency."""
        return self.readout_res_frequency - self.readout_lo_frequency


@classformatter
@dataclass(init=False, repr=True, eq=False)
class Qubit(QuantumElement):
    """A class for a generic Qubit."""

    def __init__(
        self,
        uid: str = None,
        signals: Dict[str, LogicalSignal] = None,
        parameters: Optional[Union[QubitParameters, Dict[str, Any]]] = None,
    ):
        """
        Initializes a new Qubit.

        Args:
            uid: A unique identifier for the Qubit.
            signals: A mapping of logical signals associated with the qubit.
                Qubit accepts the following keys in the mapping: 'drive', 'measure', 'acquire', 'flux'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        if isinstance(parameters, dict):
            parameters = QubitParameters(**parameters)
        if signals is None:
            signals = QuantumElementSignalMap(
                {}, key_validator=self._validate_signal_type
            )
        if isinstance(signals, dict):
            signals = QuantumElementSignalMap(
                signals, key_validator=self._validate_signal_type
            )
        super().__init__(uid, signals, parameters)

    @staticmethod
    def _validate_signal_type(name: str) -> str:
        try:
            SignalType(name)
            return name
        except ValueError:
            raise LabOneQException(
                f"Signal {name} is not one of {[enum.value for enum in SignalType]}"
            )

    @classmethod
    def from_logical_signal_group(
        cls,
        uid: str,
        lsg,
        parameters: Optional[Union[QubitParameters, Dict[str, Any]]] = None,
    ) -> "Qubit":
        """Qubit from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Qubit understands the following signal line names:

                    - drive: 'drive', 'drive_line'
                    - measure: 'measure', 'measure_line'
                    - acquire: 'acquire', 'acquire_line'
                    - flux: 'flux', 'flux_line'

                This is so that the Qubit parameters are assigned into the correct signal lines in
                calibration.
            parameters: Parameters associated with the qubit.
        """
        signal_map = {}
        for name, sig in lsg.logical_signals.items():
            sig_type = name
            if name in ["drive", "drive_line"]:
                sig_type = SignalType.DRIVE.value
            if name in ["measure", "measure_line"]:
                sig_type = SignalType.MEASURE.value
            if name in ["acquire", "acquire_line"]:
                sig_type = SignalType.ACQUIRE.value
            if name in ["flux", "flux_line"]:
                sig_type = SignalType.FLUX.value
            signal_map[sig_type] = cls._resolve_to_logical_signal_uid(sig)
        return cls(
            uid=uid, signals=QuantumElementSignalMap(signal_map), parameters=parameters
        )

    def calibration(self) -> Calibration:
        """Generate calibration from the parameters and attached signal lines.

        Returns:
            Prefilled calibration object from Qubit parameters.
        """
        calibs = {}
        if "drive" in self.signals:
            calibs[self.signals["drive"]] = SignalCalibration(
                oscillator=Oscillator(
                    uid=f"{self.uid}_drive_osc",
                    frequency=self.parameters.drive_frequency,
                )
            )
        if "measure" in self.signals:
            calibs[self.signals["measure"]] = SignalCalibration(
                oscillator=Oscillator(
                    uid=f"{self.uid}_measure_osc",
                    frequency=self.parameters.readout_frequency,
                )
            )
        if "acquire" in self.signals:
            calibs[self.signals["acquire"]] = SignalCalibration(
                oscillator=Oscillator(
                    uid=f"{self.uid}_acquire_osc",
                    frequency=self.parameters.readout_frequency,
                )
            )
        if "flux" in self.signals:
            calibs[self.signals["flux"]] = SignalCalibration()
        return Calibration(calibs)

    def experiment_signals(
        self, with_types=False
    ) -> Union[List[ExperimentSignal], List[Tuple[SignalType, ExperimentSignal]]]:
        """Experiment signals of the quantum element.

        Args:
            with_types: Return a list of tuples which consist of an mapped logical signal type and an experiment signal.
        """
        exp_signals = super().experiment_signals()
        if with_types:
            sigs = []
            for exp_sig in exp_signals:
                for role, signal in self.signals.items():
                    if signal == exp_sig.mapped_logical_signal_path:
                        role = SignalType(role)
                        sigs.append((role, exp_sig))
                        break
            return sigs
        return exp_signals
