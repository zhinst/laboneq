# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import Any, Dict

from .io_units import LogicalSignal
from .logical_signal_group import LogicalSignalGroup


@dataclass(init=False, repr=True, order=True)
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
            logical_signals: A dictionary of logical signals associated with the quantum element.
            logical_signal_group: A logical signal group associated with the quantum element.
            parameters: A dictionary of parameters associated with the quantum element.
        """
        self.uid = uid
        self.signals = {} if signals is None else signals

        self._parameters = {} if parameters is None else parameters

        if logical_signal_group is not None:
            if signals:
                raise ValueError("Cannot have both signals and logical signal_group")
            else:
                self.signals = self._parse_signals(logical_signal_group)

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

    def get_signal(self, signal_name: str, device_setup) -> LogicalSignal:
        """
        Retrieves a logical signal from the quantum element.

        Args:
            signal_name: The name of the logical signal to retrieve.
            device_setup: The device setup object containing the logical signal.

        Returns:
            The logical signal object associated with the specified name.
        """
        signal = self.signals.get(signal_name)
        group, signal = signal.split("/", 1)
        ls = device_setup.logical_signal_groups.get(group).logical_signals[signal]
        return ls

    def set_signal_group(self, logical_signal_group: LogicalSignalGroup):
        """
        Sets the logical signal group for the quantum element.

        Args:
            logical_signal_group: The logical signal group to set for the quantum element.
        """
        self.signals.update(self._parse_signal(logical_signal_group))

    @property
    def parameters(self):
        return copy(self._parameters)

    def set_parameters(self, parameters: Dict[str, Any]):
        """
        Sets the parameters for the quantum element.
        Allowed datatypes for the parameters are the following: Integer, Boolean, Float, Complex numbers,
        Numpy arrays of the above, Strings, Dictionaries of the above, LabOne Q types and None.

        Args:
            parameters: A dictionary of parameters to set for the quantum element.
        """
        self._parameters.update(parameters)

    @staticmethod
    def load(filename):
        """
        Loads a QuantumElement object from a JSON file.

        Args:
            filename: The name of the JSON file to load the QuantumElement object from.
        """
        from laboneq.dsl.serialization import Serializer

        return Serializer.from_json_file(filename, QuantumElement)

    def save(self, filename):
        """
        Save a QuantumElement object to a JSON file.

        Args:
            filename: The name of the JSON file to save the QuantumElement object.
        """
        from laboneq.dsl.serialization import Serializer

        Serializer.to_json_file(self, filename)


@dataclass(init=False, repr=True, order=True)
class Qubit(QuantumElement):
    """A class for generic qubits."""

    ...
