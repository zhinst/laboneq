# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import os
import uuid
from abc import ABC, abstractmethod
from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Tuple, Union

from laboneq.core.exceptions import LabOneQException
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.calibration import Calibration
from laboneq.dsl.device import LogicalSignalGroup
from laboneq.dsl.device.io_units import LogicalSignal
from laboneq.dsl.experiment import ExperimentSignal
from laboneq.dsl.serialization import Serializer


class SignalType(Enum):
    DRIVE = "drive"
    DRIVE_EF = "drive_ef"
    MEASURE = "measure"
    ACQUIRE = "acquire"
    FLUX = "flux"


class QuantumElementSignalMap(MutableMapping):
    def __init__(
        self,
        items: dict[str, str],
        key_validator: Callable[[str], SignalType] | None = None,
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
                key = self._key_validator(k).value
                self._items[key] = v
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

    def roles(self) -> dict[str, SignalType]:
        """Roles of the signals."""
        roles = {}
        for sig in self._items:
            roles[sig] = SignalType(sig)
        return roles


@classformatter
@dataclass(init=False, repr=True)
class QuantumElement(ABC):
    """An abstract base class for quantum elements like Qubits or tunable couplers etc."""

    uid: str
    signals: Dict[str, str]

    def __init__(
        self,
        uid: str | None = None,
        signals: dict[str, LogicalSignal] | None = None,
    ):
        """
        Initializes a new QuantumElement object.

        Args:
            uid: A unique identifier for the quantum element.
            signals: A dictionary of logical signals associated with the quantum element.
        """
        self.uid = uuid.uuid4().hex if uid is None else uid
        if signals is None:
            self.signals = QuantumElementSignalMap(
                {}, key_validator=self._coerce_signal_type
            )
        elif isinstance(signals, dict):
            sigs = {
                k: self._resolve_to_logical_signal_uid(v) for k, v in signals.items()
            }
            self.signals = QuantumElementSignalMap(
                sigs, key_validator=self._coerce_signal_type
            )
        else:
            self.signals = signals

    def __hash__(self):
        return hash(self.uid)

    @staticmethod
    def _resolve_to_logical_signal_uid(signal: Union[str, LogicalSignal]) -> str:
        return signal.path if isinstance(signal, LogicalSignal) else signal

    @staticmethod
    def _coerce_signal_type(name: str) -> SignalType:
        """Coerse signal type.

        The method can be overwritten if the child class
        does additional validation or uses custom `SignalType`.

        Raises:
            LabOneQException: If `name` is not one of `SignalType` values.
        """
        try:
            return SignalType(name)
        except ValueError:
            raise LabOneQException(
                f"Signal {name} is not one of {[enum.value for enum in SignalType]}"
            ) from None

    @classmethod
    def _from_logical_signal_group(
        cls,
        uid: str,
        lsg: LogicalSignalGroup,
        parameters: dict[str, Any] | None = None,
        signal_type_map: dict[SignalType, List[str]] | None = None,
    ) -> "QuantumElement":
        """Quantum Element from logical signal group.

        Args:
            uid: A unique identifier for the Qubit.
            lsg: Logical signal group.
                Accepted names for logical signals depend on the qubit class used
            parameters: Parameters associated with the qubit.
            signal_types: a mapping between accepted logical signal names and SignalTypes
        """
        signal_map = {}
        for name, signal in lsg.logical_signals.items():
            signal_value = name
            for signal_type, id_list in signal_type_map.items():
                if name in id_list:
                    signal_value = signal_type.value
            signal_map[signal_value] = cls._resolve_to_logical_signal_uid(signal)
        return cls(
            uid=uid, signals=QuantumElementSignalMap(signal_map), parameters=parameters
        )

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

    def experiment_signals(
        self,
        with_types: bool = False,
        with_calibration: bool = False,
    ) -> Union[List[ExperimentSignal], List[Tuple[SignalType, ExperimentSignal]]]:
        """Experiment signals of the quantum element.

        Args:
            with_types:
                When true, return a list of tuples which consist of a mapped logical signal
                type and an experiment signal. Otherwise, just return the experiment signals.
            with_calibration:
                Apply the qubit's calibration to the ExperimentSignal.
        """
        if not with_calibration:
            exp_signals = [
                ExperimentSignal(uid=k, map_to=k) for k in self.signals.values()
            ]
        else:
            exp_signals = [
                ExperimentSignal(uid=k, calibration=v, map_to=k)
                for k, v in self.calibration().items()
            ]
        if with_types:
            sigs = []
            roles = self.signals.roles()
            for exp_sig in exp_signals:
                for name, signal in self.signals.items():
                    if signal == exp_sig.mapped_logical_signal_path:
                        sigs.append((roles[name], exp_sig))
                        break
            return sigs
        return exp_signals
