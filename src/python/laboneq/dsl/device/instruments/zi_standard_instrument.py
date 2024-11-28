# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from laboneq.core.types.enums import ReferenceClockSource
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from ..instrument import Instrument


@classformatter
@dataclass(init=True, repr=True, order=True)
class ZIStandardInstrument(Instrument):
    """Base class representing a ZI instrument controlled via a LabOne Data Server.

    Attributes:
        server_uid: Dataserver UID to which this instrument is connected to.
        address: Instrument address.
        device_options: Options of the instruments.
        reference_clock_source: Reference clock source.
            Options: 'internal', 'external'
    """

    server_uid: str | None = field(default=None)
    address: str | None = field(default=None)
    device_options: str | None = field(default=None)
    reference_clock_source: ReferenceClockSource | str | None = field(default=None)

    def __post_init__(self):
        if isinstance(self.reference_clock_source, str):
            self.reference_clock_source = ReferenceClockSource(
                self.reference_clock_source.lower()
            )

    #: Device class
    device_class: int = field(default=0x0)

    def calc_options(self):
        return {
            **super().calc_options(),
            "serial": self.address,
            "interface": self.interface,
            "device_options": self.device_options,
        }
