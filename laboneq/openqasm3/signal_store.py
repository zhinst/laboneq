# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple


class SignalLineType(Enum):
    """An enum for the different signal line types."""

    MEASURE = "measure"
    CONTROL = "flux"
    ACQUIRE = "acquire"
    DRIVE = "drive"


@dataclass
class Signal:
    signal_type: SignalLineType
    exp_signal: str


class SignalStore:
    def __init__(self, exp_map):
        self._exp_map: Dict[str, str] = deepcopy(exp_map)
        self.user_map: Dict[str, List[Signal]] = {}

    def leftover_raise(self):
        """Raise an exception if not all experiment signals have been used up."""
        if self._exp_map:
            raise ValueError(f"Unassigned experiment signals: {self._exp_map.keys()}")

    def register_signal_group(
        self,
        qubit: str,
        signals: List[Tuple[SignalLineType, str]],
    ):
        """Register the LabOne Q logical signal group belonging to a QASM qubit."""
        for pair in signals:
            suggested_signal = pair[1]
            try:
                self._exp_map.pop(suggested_signal)
            except KeyError as e:
                raise ValueError(
                    f"Invalid signal: '{suggested_signal}' not found in experiment signals or already used.",
                ) from e
        self.user_map[qubit] = [
            Signal(signal_type, exp_signal) for (signal_type, exp_signal) in signals
        ]
