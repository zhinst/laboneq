# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A connection builder for instruments in a device setup."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import attrs

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import PhysicalChannelType
from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device.connection import InternalConnection, SignalConnection

if TYPE_CHECKING:
    from laboneq.dsl.device.instrument import Instrument
    from laboneq.dsl.enums import IOSignalType


class DeviceSetupInternalException(LabOneQException):
    """Raised when a device setup cannot be built as declared."""


_PHYSICAL_CHANNEL_TYPES = {
    "iq": PhysicalChannelType.IQ_CHANNEL,
    "acquire": PhysicalChannelType.IQ_CHANNEL,
    "rf": PhysicalChannelType.RF_CHANNEL,
}


def physical_channel_type(signal_type: str) -> PhysicalChannelType:
    """The physical channel a signal of this type forms.

    Args:
        signal_type: Signal type as written in the descriptor: `iq`, `acquire` or `rf`

    Raises:
        ValueError: The signal type is not one of `iq`, `acquire` or `rf`.
    """
    try:
        return _PHYSICAL_CHANNEL_TYPES[signal_type]
    except KeyError:
        raise ValueError(f"Invalid channel type: '{signal_type}'") from None


@attrs.define(frozen=True)
class ResolvedPort:
    """One local port a connection occupies, and what it carries."""

    #: UID of the port on the instrument.
    port: str

    #: Signal the port carries.
    signal_type: IOSignalType


@attrs.define(frozen=True)
class ResolvedConnection:
    """The instrument ports and physical channel a declared connection maps to."""

    #: Ports the connection occupies.
    ports: list[ResolvedPort] = attrs.field(factory=list)

    #: Direction of the connection.
    direction: IODirection = IODirection.OUT

    #: Connection physical channel type, or `None` if they form none. A
    #: connection that forms no physical channel makes no logical signal
    #: either.
    channel_type: PhysicalChannelType | None = None


#: Makes ports for an instrument given the instrument and the connection.
#: The resolved connection include the instrument ports and additional
#: information about the physical channel that is created.
ConnectionBuilder = Callable[
    ["Instrument", InternalConnection | SignalConnection], ResolvedConnection
]
