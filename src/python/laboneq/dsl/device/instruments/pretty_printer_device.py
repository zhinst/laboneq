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
        self._ports: list[Port] = []

    def calc_options(self):
        return {
            **super().calc_options(),
        }

    def append_port(self, port_name: str, signal_type_str: str):
        if signal_type_str == "rf":
            direction = IODirection.OUT
            signal_type = IOSignalType.RF
        elif signal_type_str == "iq":
            direction = IODirection.OUT
            signal_type = IOSignalType.IQ
        elif signal_type_str == "acquire":
            direction = IODirection.IN
            signal_type = IOSignalType.IQ
        else:
            raise AssertionError("invalid signal type")
        self._ports.append(
            Port(
                direction=direction,
                uid=port_name,
                signal_type=signal_type,
                physical_port_ids=[str(0)],
                connector_labels=[f"Wave {0 + 1}"],
            )
        )

    @property
    def ports(self) -> list[Port]:
        return self._ports
