# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

from laboneq.dsl.parameter import Parameter

amplifier_pump_id = 0


def amplifier_pump_id_generator():
    global amplifier_pump_id
    retval = f"ap{amplifier_pump_id}"
    amplifier_pump_id += 1
    return retval


@dataclass(init=True, repr=True, order=True)
class AmplifierPump:
    """Data object containing settings for the Parametric Pump Controller."""

    #: Unique identifier. If left blank, a new unique ID will be generated.
    uid: str = field(default_factory=amplifier_pump_id_generator)

    pump_freq: Optional[Union[float, Parameter]] = None
    pump_power: Optional[Union[float, Parameter]] = None
    cancellation: bool = True
    alc_engaged: bool = True
    use_probe: bool = False
    probe_frequency: Optional[Union[float, Parameter]] = None
    probe_power: Optional[Union[float, Parameter]] = None
