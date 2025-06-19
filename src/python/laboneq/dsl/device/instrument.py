# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.device.ports import Port

from .connection import Connection


@classformatter
@attrs.define(slots=False)
class Instrument:
    """Class representing an instrument."""

    #: Unique identifier.
    uid: str = attrs.field(default=None)

    #: Interface of this instrument. The default is 1GbE (1 Gbit ethernet)
    interface: str = attrs.field(default="1GbE")

    #: Connections of this instrument.
    connections: list[Connection] = attrs.field(factory=list)

    def output_by_uid(self, uid) -> Port | None:
        for o in self.ports:
            if o.uid == uid and o.direction == IODirection.OUT:
                return o
        return None

    def input_by_uid(self, uid) -> Port | None:
        for i in self.ports:
            if i.uid == uid and i.direction == IODirection.IN:
                return i
        return None

    def calc_options(self):
        return {}

    def calc_driver(self):
        return self.__class__.__name__

    @property
    def ports(self) -> list[Port]:
        """Input and output ports that are part of this instrument."""
        return []
