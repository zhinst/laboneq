# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.device import Port
from laboneq.dsl.enums import IOSignalType

from .zi_standard_instrument import ZIStandardInstrument


@classformatter
@dataclass(init=True, repr=True, order=True)
class HDAWG(ZIStandardInstrument):
    """Class representing a ZI HDAWG instrument."""

    @property
    def ports(self):
        """Input and output ports that are part of this instrument."""
        inputs = [
            Port(
                IODirection.IN,
                uid="ZSYNCS/0",
                signal_type=IOSignalType.ZSYNC,
                physical_port_ids=["0"],
                connector_labels=["ZSync"],
            )
        ]

        outputs = []

        for i in range(8):
            outputs += [
                Port(
                    IODirection.OUT,
                    uid=f"SIGOUTS/{i}",
                    signal_type=IOSignalType.RF,
                    physical_port_ids=[str(i)],
                    connector_labels=[f"Wave {i + 1}"],
                ),
            ]
        outputs.append(
            Port(
                IODirection.OUT,
                uid="DIOS/0",
                signal_type=IOSignalType.DIO,
                physical_port_ids=["0"],
                connector_labels=["DIO 32bit / Marker"],
            )
        )
        return inputs + outputs
