# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Union

from laboneq.dsl.enums import CarrierType, ModulationType

if TYPE_CHECKING:
    from laboneq.dsl import Parameter

oscillator_id = 0


def oscillator_uid_generator():
    global oscillator_id
    retval = f"osc_{oscillator_id}"
    oscillator_id += 1
    return retval


@dataclass(init=True, repr=True, order=True)
class Oscillator:
    """
    This oscillator class represents an oscillator on a `PhysicalChannel`.
    All pulses played on any signal line attached to this physcical channel will be modulated with the oscillator assigned to that channel.

    Args:
        frequency (float): The frequency in units of Hz
        modulation_type (ModulationType): The modulation type (`ModulationType.SOFTWARE` or `ModulationType.HARDWARE`).
            When choosing a HARDWARE oscillator, a digital oscillator on the instrument will be used to modulate the output signal,
            while the choice SOFTWARE will lead to waveform being modulated in software before upload to the instruments.
            The default `ModulationType.AUTO` currently falls back to `ModulationType.Software`.
        carrier_type (CarrierType): The carrier type, defaults to radio frequency (`CarrierType.RF`)
    """

    uid: str = field(default_factory=oscillator_uid_generator)
    frequency: Union[float, Parameter] = field(default=None)
    modulation_type: ModulationType = field(default=ModulationType.AUTO)
    carrier_type: CarrierType = field(default=CarrierType.RF)

    def __hash__(self):
        return hash(self.uid)
