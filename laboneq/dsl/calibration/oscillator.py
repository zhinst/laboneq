# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union, TYPE_CHECKING

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
    uid: str = field(default_factory=oscillator_uid_generator)
    carrier_type: CarrierType = field(default=CarrierType.RF)
    frequency: Union[float, Parameter] = field(default=None)
    modulation_type: ModulationType = field(default=ModulationType.AUTO)

    def __hash__(self):
        return hash(self.uid)
