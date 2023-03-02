# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device import Port

from ...enums import IOSignalType
from .zi_standard_instrument import ZIStandardInstrument


@dataclass(init=True, repr=True, order=True)
class UHFQA(ZIStandardInstrument):
    """Class representing a ZI UHFQA instrument."""

    @property
    def ports(self):
        """Input and output ports that are part of this instrument."""
        inputs = [
            Port(
                IODirection.IN,
                uid="DIOS/0",
                signal_type=IOSignalType.DIO,
                physical_port_ids=["0"],
                connector_labels=["DIO"],
            )
        ]
        for i in range(10):
            inputs.append(
                Port(
                    IODirection.IN,
                    uid=f"QAS/{i}",
                    signal_type=IOSignalType.IQ,
                    physical_port_ids=["0", "1"],
                    connector_labels=["Signal Input 1", "Signal Input 2"],
                )
            )

        outputs = [
            Port(
                IODirection.OUT,
                uid="SIGOUTS/0",
                signal_type=IOSignalType.RF,
                physical_port_ids=["0"],
                connector_labels=["Signal Output 1"],
            ),
            Port(
                IODirection.OUT,
                uid="SIGOUTS/1",
                signal_type=IOSignalType.RF,
                physical_port_ids=["1"],
                connector_labels=["Signal Output 2"],
            ),
        ]

        for i in range(10):
            outputs.append(
                Port(
                    IODirection.OUT,
                    uid=f"QAS/{i}/RESULT/0",
                    signal_type=IOSignalType.I,
                    physical_port_ids=[],
                    connector_labels=[],
                )
            )
            outputs.append(
                Port(
                    IODirection.OUT,
                    uid=f"QAS/{i}/RESULT/1",
                    signal_type=IOSignalType.Q,
                    physical_port_ids=[],
                    connector_labels=[],
                )
            )

        return inputs + outputs
