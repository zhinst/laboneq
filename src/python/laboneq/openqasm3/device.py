# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass


@dataclass
class Port:
    """A port on a device.

    Attributes:
        qubit: Qubit UID.
        signal: Signal line name.
    """

    qubit: str
    signal: str


def port(qubit: str, signal: str) -> Port:
    """Create a port to qubit signal line.

    Arguments:
        qubit: UID of the qubit.
        signal: Name of the signal line.

    Returns:
        A port.
    """
    return Port(qubit, signal)
