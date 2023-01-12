# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import typing
from dataclasses import dataclass, field
from typing import List

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device.ports import Port

from .connection import Connection


@dataclass(init=True, repr=True, order=True)
class Instrument:
    """Class representing an instrument."""

    #: Unique identifier.
    uid: str = field(default=None)

    #: Interface of this instrument. The default is 1GbE (1 Gbit ethernet)
    interface: str = field(default="1GbE")

    #: Connections of this instrument.
    connections: typing.List[Connection] = field(default_factory=list)

    def output_by_uid(self, uid):
        for o in self.ports:
            if o.uid == uid and o.direction == IODirection.OUT:
                return o
        return None

    def input_by_uid(self, uid):
        for i in self.ports:
            if i.uid == uid and i.direction == IODirection.IN:
                return i
        return None

    def calc_options(self):
        return {}

    def calc_driver(self):
        return self.__class__.__name__

    @property
    def ports(self) -> List[Port]:
        """Input and output ports that are part of this instrument."""
        return []
