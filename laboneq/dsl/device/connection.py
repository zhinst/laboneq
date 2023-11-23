# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional

from laboneq.core import path as qct_path
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.enums import IODirection, IOSignalType


@classformatter
@dataclass(init=True, repr=True, order=True)
class Connection:
    direction: IODirection = field(default=IODirection.OUT)
    local_path: str | None = field(default=None)
    local_port: str | None = field(default=None)
    remote_path: str | None = field(default=None)
    remote_port: str | None = field(default=None)
    signal_type: Optional[IOSignalType] = field(default=None)


@dataclass
class SignalConnection:
    """Connection to a logical signal

    Attributes:
        uid: Logical signal name in a format  '<group>/<name>'
        type: Type of the signal
            Options: iq, rf, acquire
        ports: List of target instrument ports the signal is connected to.
    """

    uid: str
    # Optional: inferred from device and ports
    type: str = field(default=None)
    ports: list[str] = field(default_factory=list)

    def __post_init__(self):
        parts = self.uid.split(qct_path.Separator)
        if len(parts) != 2 or (not parts[0] and parts[1]):
            raise ValueError(
                f"Signal connection uid must be of format <group>{qct_path.Separator}<name>"
            )

    @property
    def group(self):
        return self.uid.split(qct_path.Separator)[0]

    @property
    def name(self):
        return self.uid.split(qct_path.Separator)[1]


@dataclass
class InternalConnection:
    """Setup internal connection between Zurich Instrument devices.

    Attributes:
        to: UID of the instrument or signal to which the device is connected to.
        from_port: From which port the connection is made in the source device.
    """

    to: str
    from_port: str

    def __post_init__(self):
        if not isinstance(self.from_port, str) and self.from_port is not None:
            if len(self.from_port) != 1:
                raise ValueError("To instrument connection takes only one port.")
            self.from_port = self.from_port[0]

    @property
    def ports(self) -> list[str]:
        return [self.from_port]


def create_connection(
    ports: Iterable[str] | str | None = None, **kwargs
) -> SignalConnection | InternalConnection:
    """Create a connection.

    Args:
        ports: Ports of the target instrument.

    Keyword Args:
        Only one of the following can be exists:

            - to_instrument: If the connection is to an another instrument
                Takes one instrument port.

            - to_signal: If the connection is to a signal
                The number of allowed ports depends on the instrument and its port.

    Returns:
        Created connection.

    Raises:
        ValueError: Neither or both 'to_instrument' and 'to_signal' defined.
            'to_instrument' was given more than one port.
            'to_signal' input format is invalid.

    !!! version-added "Added in version 2.19.0"
    """
    if "to_instrument" in kwargs and "to_signal" in kwargs:
        raise ValueError(
            "Get both 'to_instrument' and 'to_signal'. Use only either one of them."
        )
    if "to_instrument" in kwargs:
        to = kwargs["to_instrument"]
        return InternalConnection(to, from_port=ports)
    if "to_signal" in kwargs:
        to = kwargs["to_signal"]
        if ports is None:
            ports = []
        elif isinstance(ports, str):
            ports = [ports]
        return SignalConnection(uid=to, ports=ports, type=None)
    raise ValueError("Either 'to_instrument' or 'to_signal' must be defined.")
