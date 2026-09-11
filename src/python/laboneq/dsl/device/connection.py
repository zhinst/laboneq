# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Iterable, Optional

import attrs

from laboneq.core import path as qct_path
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.device.io_units.logical_signal import (
    LogicalSignal,
    resolve_logical_signal_ref,
)
from laboneq.dsl.enums import IODirection

if TYPE_CHECKING:
    from laboneq.dsl.device.io_units.logical_signal import LogicalSignalRef
    from laboneq.dsl.enums import IOSignalType


@classformatter
@attrs.define
class Connection:
    direction: IODirection = attrs.field(default=IODirection.OUT)
    local_port: str | None = attrs.field(default=None)
    remote_path: str | None = attrs.field(default=None)
    signal_type: Optional[IOSignalType] = attrs.field(default=None)


@classformatter
@attrs.define
class SignalConnection:
    """Connection to a logical signal

    Attributes:
        uid: Logical signal name in a format  '<group>/<name>'
        type: Type of the signal
            Options: iq, rf, acquire
        ports: List of target instrument ports the signal is connected to.
            A single already-defined logical signal (as its uid/path string
            or a `LogicalSignal` object) may be given instead of a list of
            raw ports; it then resolves to the raw port(s) already backing
            that signal.
    """

    uid: str
    # Optional: inferred from device and ports
    type: str | None = attrs.field(default=None)
    ports: list[str] = attrs.field(factory=list)

    def __attrs_post_init__(self):
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


@classformatter
@attrs.define(init=False)
class InternalConnection:
    """Setup internal connection between Zurich Instrument devices.

    Attributes:
        to: UID of the instrument or signal to which the device is connected to.
        from_port: From which port the connection is made in the source device.
    """

    to: str
    from_port: str | None

    def __init__(self, to: str, from_port: str | Iterable[str] | None = None):
        self.to = to
        if isinstance(from_port, Iterable) and not isinstance(from_port, str):
            from_port_iter = iter(from_port)
            self.from_port = next(from_port_iter, None)
            if next(from_port_iter, None) is not None:
                raise ValueError("To instrument connection takes zero or one port.")
        else:
            self.from_port = from_port

    @property
    def ports(self) -> list[str]:
        return [] if self.from_port is None else [self.from_port]


def create_connection(
    ports: Iterable[str] | str | LogicalSignalRef | None = None, **kwargs
) -> SignalConnection | InternalConnection:
    """Create a connection.

    Args:
        ports: Ports of the target instrument. A single already-defined
            logical signal (as its uid/path string or a `LogicalSignal`
            object) may be given instead, in which case the connection
            resolves to the raw port(s) already backing that signal.

    Keyword Args:
        Only one of the following can exist:

            - to_instrument: If the connection is to another instrument
                Takes one instrument port.

            - to_signal: If the connection is to a signal
                The number of allowed ports depends on the instrument and its port.

        type: Optional argument, can be used to specify a signal type, allowed values are "iq", "rf" or "acquire"

    Returns:
        Created connection.

    Raises:
        ValueError:
         - Either none or both 'to_instrument' and 'to_signal' defined.
         - 'to_instrument' was given more than one port.
         - 'to_signal' input format is invalid.
    """
    if "to_instrument" in kwargs and "to_signal" in kwargs:
        raise ValueError(
            "Get both 'to_instrument' and 'to_signal'. Use only either one of them."
        )
    if "to_instrument" in kwargs:
        to = kwargs["to_instrument"]
        if isinstance(ports, LogicalSignal):
            raise ValueError(
                "A LogicalSignal reference for 'ports' is only supported "
                "when connecting to a signal ('to_signal'), not to an "
                "instrument ('to_instrument')."
            )
        return InternalConnection(to, from_port=ports)
    if "to_signal" in kwargs:
        to = kwargs["to_signal"]
        if ports is None:
            ports = []
        elif isinstance(ports, str):
            ports = [ports]
        elif isinstance(ports, LogicalSignal):
            ports = [resolve_logical_signal_ref(ports)]
        elif isinstance(ports, Iterable):
            for p in ports:
                if isinstance(p, LogicalSignal):
                    raise ValueError(
                        "Only a single LogicalSignal is allowed for 'ports'."
                    )
        if "signal_type" in kwargs:
            warnings.warn(
                "Argument 'signal_type' of 'create_connection' has been renamed to 'type'.",
                FutureWarning,
                stacklevel=2,
            )
            kwargs["type"] = kwargs["signal_type"]
            del kwargs["signal_type"]
        if "type" in kwargs:
            signal_type = kwargs["type"]
        else:
            signal_type = None
        return SignalConnection(uid=to, ports=list(ports), type=signal_type)
    raise ValueError("Either 'to_instrument' or 'to_signal' must be defined.")
