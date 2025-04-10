# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs

from laboneq.core.types.enums import ReferenceClockSource
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from ..instrument import Instrument


def _reference_clock_source_converter(
    value: ReferenceClockSource | str | None,
) -> ReferenceClockSource | None:
    if isinstance(value, str):
        value = ReferenceClockSource(value.lower())
    return value


@classformatter
@attrs.define(slots=False)
class ZIStandardInstrument(Instrument):
    """Base class representing a ZI instrument controlled via a LabOne Data Server.

    Attributes:
        server_uid: Dataserver UID to which this instrument is connected to.
        address: Instrument address.
        device_options: Options of the instruments.
        reference_clock_source: Reference clock source.
            Options: 'internal', 'external'
    """

    server_uid: str | None = attrs.field(default=None)
    address: str | None = attrs.field(default=None)
    device_options: str | None = attrs.field(default=None)
    reference_clock_source: ReferenceClockSource | None = attrs.field(
        default=None, converter=_reference_clock_source_converter
    )

    #: Device class
    device_class: int = attrs.field(default=0x0)

    def calc_options(self):
        return {
            **super().calc_options(),
            "serial": self.address,
            "interface": self.interface,
            "device_options": self.device_options,
        }
