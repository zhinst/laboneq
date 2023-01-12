# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device import Port
from laboneq.dsl.enums import IOSignalType

from .zi_standard_instrument import ZIStandardInstrument


@dataclass(init=True, repr=True, order=True)
class PQSC(ZIStandardInstrument):
    """Class representing a ZI PQSC instrument."""

    #: The reference clock frequency
    reference_clock: float = field(default=None)

    @property
    def ports(self):
        """Input and output ports that are part of this instrument."""
        return [
            Port(
                IODirection.OUT,
                uid=f"ZSYNCS/{i}",
                signal_type=IOSignalType.ZSYNC,
                physical_port_ids=[str(i)],
                connector_labels=[f"ZSync {i + 1}"],
            )
            for i in range(18)
        ]
