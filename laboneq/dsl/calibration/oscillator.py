# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.enums import CarrierType, ModulationType

if TYPE_CHECKING:
    from laboneq.dsl import Parameter

oscillator_id = 0


def oscillator_uid_generator():
    global oscillator_id
    retval = f"osc_{oscillator_id}"
    oscillator_id += 1
    return retval


@classformatter
@dataclass(init=True, repr=True, order=True)
class Oscillator:
    """
    This oscillator class represents an oscillator on a `PhysicalChannel`.
    All pulses played on any signal line attached to this physical channel will be
    modulated with the oscillator assigned to that channel.

    Attributes:
        frequency (float):
            The oscillator frequency. Units: Hz.
        modulation_type (ModulationType):
            The modulation type (`ModulationType.SOFTWARE` or
            `ModulationType.HARDWARE`). When choosing a HARDWARE oscillator, a digital
            oscillator on the instrument will be used to modulate the output signal,
            while the choice SOFTWARE will lead to waveform being modulated in software
            before upload to the instruments.
            The default, `ModulationType.AUTO`, resolves to `HARDWARE` in most situations,
            except for QA instruments in integration mode, where it resolves to
            `SOFTWARE`.
        carrier_type (CarrierType):
            Deprecated. The carrier type is no longer used. Default: `CarrierType.RF`.

            !!! version-changed "Deprecated in 2.7"
                The `carrier_type` has no effect.

            !!! version-changed "Changed in 2.16"
                `ModulationType.AUTO` now is more sensibly resolved. Previously it would
                 always fall back to `SOFTWARE`.
    """

    uid: str = field(default_factory=oscillator_uid_generator)
    frequency: float | Parameter | None = field(default=None)
    modulation_type: ModulationType = field(default=ModulationType.AUTO)
    carrier_type: CarrierType = field(default=None)

    def __post_init__(self):
        if self.carrier_type is not None:
            warnings.warn(
                "`Oscillator` argument `carrier_type` will be removed in the future versions. It has no functionality.",
                FutureWarning,
                stacklevel=2,
            )
        else:
            self.carrier_type = CarrierType.RF

    def __hash__(self):
        return hash(self.uid)
