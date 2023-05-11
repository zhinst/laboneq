# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict, Union

from laboneq.dsl.serialization import Serializer

from .io_units import LogicalSignal
from .logical_signal_group import LogicalSignalGroup


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
        logical_signal_group: LogicalSignalGroup = None,
        parameters: Dict[str, Any] = None,
    ):
        """
        Initializes a new QuantumElement object.

        Args:
            uid: A unique identifier for the quantum element.
            signals: A dictionary of logical signals associated with the quantum element.
            logical_signal_group: A logical signal group associated with the quantum element.
            parameters: A dictionary of parameters associated with the quantum element.
        """
        self.uid = uuid.uuid4().hex if uid is None else uid
        self.signals = (
            {
                k: (v.uid if isinstance(v, LogicalSignal) else v)
                for k, v in signals.items()
            }
            if signals
            else {}
        )
        self._parameters = {} if parameters is None else parameters
        if logical_signal_group is not None:
            if signals:
                raise ValueError("Cannot have both signals and logical signal_group")
            else:
                self.signals = self._parse_signals(logical_signal_group)

    def __hash__(self):
        return hash(self.uid)

    @property
    def parameters(self):
        return copy(self._parameters)

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

    def _parse_signals(
        self, logical_signal_group: LogicalSignalGroup
    ) -> Dict[str, str]:
        return {k: v.uid for (k, v) in logical_signal_group.logical_signals.items()}

    def add_signals(self, signals: Dict[str, LogicalSignal]):
        """
        Adds logical signals to the quantum element.

        Args:
            signals: A dictionary of logical signals to add to the quantum element.
        """
        self.signals.update({k: v.uid for (k, v) in signals.items()})

    def set_signal_group(self, logical_signal_group: LogicalSignalGroup):
        """
        Sets the logical signal group for the quantum element.

        Args:
            logical_signal_group: The logical signal group to set for the quantum element.
        """
        self.signals.update(self._parse_signals(logical_signal_group))

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Sets the parameters for the quantum element.
        Allowed datatypes for the parameters are the following: Integer, Boolean, Float, Complex numbers,
        Numpy arrays of the above, Strings, Dictionaries of the above, LabOne Q types and None.

        Args:
            parameters: A dictionary of parameters to set for the quantum element.
        """
        self._parameters.update(parameters)

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


@dataclass(init=False, repr=True, eq=False)
class Qubit(QuantumElement):
    """A class for generic qubits."""

    ...
