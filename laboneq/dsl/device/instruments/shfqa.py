# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device import Port

from ...enums import IOSignalType
from .zi_standard_instrument import ZIStandardInstrument


@dataclass(init=True, repr=True, order=True)
class SHFQA(ZIStandardInstrument):
    """Class representing a ZI SHFQA instrument."""

    is_qc: bool = False

    def calc_options(self):
        return {**super().calc_options(), "is_qc": self.is_qc}

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
            ),
            Port(
                IODirection.IN,
                uid="ZSYNCS/0",
                signal_type=IOSignalType.ZSYNC,
                physical_port_ids=["0"],
                connector_labels=["ZSync"],
            ),
        ]
        for ch in range(4):
            inputs.append(
                Port(
                    IODirection.IN,
                    uid=f"QACHANNELS/{ch}/INPUT",
                    signal_type=IOSignalType.IQ,
                    physical_port_ids=[f"{ch}"],
                    connector_labels=[f"Signal Input {ch+1}"],
                )
            )

        outputs = []
        for ch in range(4):
            outputs.append(
                Port(
                    IODirection.OUT,
                    uid=f"QACHANNELS/{ch}/OUTPUT",
                    signal_type=IOSignalType.IQ,
                    physical_port_ids=[f"{ch}"],
                    connector_labels=[f"Signal Output {ch+1}"],
                )
            )
        for i in range(16):
            outputs.append(
                Port(
                    IODirection.OUT,
                    uid=f"QACHANNELS/{ch}/RESULT/{i}",
                    signal_type=IOSignalType.IQ,
                    physical_port_ids=[],
                    connector_labels=[],
                )
            )
        return inputs + outputs
