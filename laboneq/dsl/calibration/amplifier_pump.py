# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.parameter import Parameter

amplifier_pump_id = 0


def amplifier_pump_id_generator():
    global amplifier_pump_id
    retval = f"ap{amplifier_pump_id}"
    amplifier_pump_id += 1
    return retval


@classformatter
@dataclass(init=True, repr=True, order=True)
class AmplifierPump:
    """Settings for the SHF Parametric Pump Controller (SHFPPC).

    Attributes:
        uid (str):
            Unique identifier. If left blank, a new unique ID will be
            generated.
        pump_freq (float | Parameter | None):
            Sets the pump frequency node. Default `None`.
        pump_power (float | Parameter | None):
            Sets the pump power node. Default `None`.
        cancellation (bool):
            Enable pump tone cancellation. Default `True`.
        alc_engaged (bool):
            Enable the automatic level control for pump
            tone output. Default `True`.
        use_probe (bool):
            Enable probe tone output. Default `False`.
        probe_frequency (float | Parameter | None):
            Sets the frequency of the generated probe tone.
            Units: Hz. Default: `None`.
        probe_power (float | Parameter | None):
            Sets the output power of the generated probe tone.
            Units: dBm. Default: `None`.

    Notes:
        If an attribute is set to `None`, the corresponding node
        is not set.
    """

    #: Unique identifier. If left blank, a new unique ID will be generated.
    uid: str = field(default_factory=amplifier_pump_id_generator)

    pump_freq: float | Parameter | None = None
    pump_power: float | Parameter | None = None
    cancellation: bool = True
    alc_engaged: bool = True
    use_probe: bool = False
    probe_frequency: float | Parameter | None = None
    probe_power: float | Parameter | None = None
