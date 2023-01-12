# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.dsl.device import Port

from ...enums import IOSignalType
from .zi_standard_instrument import ZIStandardInstrument


@dataclass(init=True, repr=True, order=True)
class SHFSG(ZIStandardInstrument):
    """Class representing a ZI SHFSG instrument."""

    is_qc: bool = False
    qc_with_qa: bool = False

    def calc_options(self):
        return {
            **super().calc_options(),
            "is_qc": self.is_qc,
            "qc_with_qa": self.qc_with_qa,
        }

    @property
    def ports(self):
        """Input and output ports that are part of this instrument."""
        retval = [
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
        retval.extend(
            [
                Port(
                    IODirection.OUT,
                    uid=f"SGCHANNELS/{ch}/OUTPUT",
                    signal_type=IOSignalType.IQ,
                    physical_port_ids=[f"{ch}"],
                    connector_labels=[f"Signal Output {ch+1}"],
                )
                for ch in range(8)
            ]
        )
        return retval
