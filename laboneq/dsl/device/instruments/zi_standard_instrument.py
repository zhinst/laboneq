# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from ..instrument import Instrument
from laboneq.core.types.enums import ReferenceClockSource


@dataclass(init=True, repr=True, order=True)
class ZIStandardInstrument(Instrument):
    """Base class representing a ZI instrument controlled via a LabOne Data Server."""

    #: Unique identifier of the server where that device is controlled from.
    server_uid: str = field(default=None)

    #: Identifier for locating this instrument.
    address: str = field(default=None)

    #: Reference clock source, None for default
    reference_clock_source: ReferenceClockSource = field(default=None)

    def calc_options(self):
        return {
            "serial": self.address,
            "interface": self.interface,
        }
