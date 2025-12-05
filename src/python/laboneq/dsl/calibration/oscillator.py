# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings

import attrs

from laboneq.core.utilities.attrs_helpers import validated_field
from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.enums import CarrierType, ModulationType
from laboneq.dsl.parameter import Parameter

oscillator_id = 0


def oscillator_uid_generator() -> str:
    global oscillator_id
    retval = f"osc_{oscillator_id}"
    oscillator_id += 1
    return retval


def _carrier_type_validator(
    _self: Oscillator, _attribute: attrs.Attribute, value: CarrierType | None
) -> None:
    if value is not None:
        warnings.warn(
            "`Oscillator` argument `carrier_type` will be removed in the future versions. It has no functionality.",
            FutureWarning,
            stacklevel=2,
        )


@classformatter
@attrs.define(slots=False)
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
            except for i) QA instruments in integration mode without LRT support
            or ii) RF channels on HDAWGs, where it resolves to `SOFTWARE`.
        carrier_type (CarrierType):
            Deprecated. The carrier type is no longer used. Default: `CarrierType.RF`.

            !!! version-changed "Deprecated in 2.7"
                The `carrier_type` has no effect.

            !!! version-changed "Changed in 2.16"
                `ModulationType.AUTO` now is more sensibly resolved. Previously it would
                 always fall back to `SOFTWARE`.

    !!! version-changed "Changed in version 26.1.0"

        The types of the attributes are now validated when an `Oscillator` instance is
        created or when an attribute is set. A `TypeError` is raised if the type of the
        supplied value is incorrect.
    """

    uid: str = validated_field(factory=oscillator_uid_generator)
    frequency: float | Parameter | None = validated_field(default=None)
    modulation_type: ModulationType = validated_field(default=ModulationType.AUTO)
    carrier_type: CarrierType | None = validated_field(
        default=None, validator=_carrier_type_validator
    )
