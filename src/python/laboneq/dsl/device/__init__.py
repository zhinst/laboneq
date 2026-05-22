# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from .connection import Connection, create_connection
from .instrument import Instrument
from .logical_signal_group import LogicalSignalGroup
from .physical_channel_group import PhysicalChannelGroup
from .ports import Port
from .server import Server

from .device_setup import DeviceSetup  # isort: skip


__all__ = [
    "Connection",
    "DeviceSetup",
    "Instrument",
    "LogicalSignalGroup",
    "PhysicalChannelGroup",
    "Port",
    "Server",
    "create_connection",
]
# The following import is required to register the system description builder for QCCS.
from . import (
    system_description_builder_qccs as _system_description_builder_qccs,  # noqa: F401
)


@runtime_checkable
class SystemDescription(Protocol):
    """Base class for system descriptions."""

    @abstractmethod
    def get_fingerprint(self) -> str:
        """Compute stable hash for hardware configuration matching.

        Returns hex fingerprint.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def uid(self) -> str:
        """Unique identifier for the device setup."""
        raise NotImplementedError

    @uid.setter
    @abstractmethod
    def uid(self, value: str) -> None:
        """Set unique identifier for the device setup."""
        raise NotImplementedError
