# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.device import Port

from ...enums import IOSignalType
from .zi_standard_instrument import ZIStandardInstrument


@classformatter
@dataclass(init=True, repr=True, order=True)
class PRETTYPRINTERDEVICE(ZIStandardInstrument):
    """Class representing a virtual instrument that pretty prints some info about the experiment."""

    def __post_init__(self):
        self.device_class = 0x1
        self._ports: list[str] = []

    def calc_options(self):
        return {
            **super().calc_options(),
        }

    @property
    def ports(self) -> list[Port]:
        return [
            Port(
                IODirection.OUT,
                uid=port,
                signal_type=IOSignalType.RF,
                physical_port_ids=[str(0)],
                connector_labels=[f"Wave {0 + 1}"],
            )
            for port in self._ports
        ]
