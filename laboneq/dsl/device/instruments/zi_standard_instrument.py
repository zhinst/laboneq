# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from laboneq.core.types.enums import ReferenceClockSource
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from ..instrument import Instrument


@classformatter
@dataclass(init=True, repr=True, order=True)
class ZIStandardInstrument(Instrument):
    """Base class representing a ZI instrument controlled via a LabOne Data Server."""

    #: Unique identifier of the server where that device is controlled from.
    server_uid: str = field(default=None)

    #: Identifier for locating this instrument.
    address: str = field(default=None)

    #: Options configured on this instrument.
    device_options: str = field(default=None)

    #: Reference clock source, None for default
    reference_clock_source: ReferenceClockSource = field(default=None)

    def calc_options(self):
        return {
            **super().calc_options(),
            "serial": self.address,
            "interface": self.interface,
            "device_options": self.device_options,
        }
