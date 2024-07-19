# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.data.calibration import CancellationSource
from laboneq.dsl.calibration.observable import Observable
from laboneq.dsl.parameter import Parameter

amplifier_pump_id = 0


def amplifier_pump_id_generator():
    global amplifier_pump_id
    retval = f"ap{amplifier_pump_id}"
    amplifier_pump_id += 1
    return retval


@classformatter
@dataclass(init=True, repr=True, order=True)
class AmplifierPump(Observable):
    """Settings for the SHF Parametric Pump Controller (SHFPPC).

    Attributes:
        uid (str):
            Unique identifier. If left blank, a new unique ID will be
            generated.
        pump_frequency (float | Parameter | None):
            Set the pump frequency node. Default `None`.
        pump_power (float | Parameter | None):
            Set the pump power node. Units: dBm. Default `None`.
        pump_on (bool):
            Enable the pump tone. Default `True`.
        pump_filter_on (bool):
            Enable the integrated low-pass filter for the pump tone. Default: `True`.
        cancellation_on (bool):
            Enable pump tone cancellation. Default `True`.
        cancellation_phase (float | Parameter | None):
            Set the phase shift of the cancellation tone. Units: radians. Default `None`.
        cancellation_attenuation (float | Parameter | None):
            Set the attenuation of the cancellation tone. Positive values _reduce_ the
            cancellation tone power. Default `None`.
        cancellation_source (CancellationSource):
            Set the source of the cancellation tone. Default: internal.
        cancellation_source_frequency (float | None):
            Specify the cancellation tone frequency when using the *external*
            cancellation tone generator. Leave at `None` when supplying the
            cancellation tone internally (the frequency then matches that of the
            pump tone).
        alc_on (bool):
            Enable the automatic level control for pump tone output. Default `True`.
        probe_on (bool):
            Enable probe tone output. Default `False`.
        probe_frequency (float | Parameter | None):
            Set the frequency of the generated probe tone. Required if `probe_on` is
            `True`. Units: Hz. Default: `None`.
        probe_power (float | Parameter | None):
            Set the output power of the generated probe tone.
            Units: dBm. Default: `None`.

    Notes:
        If an attribute is set to `None`, the corresponding node is not set.

    !!! version-changed "Some fields were renamed in version 2.24.0"

        - `AmplifierPump.pump_freq` is now `AmplifierPump.pump_frequency`
        - `AmplifierPump.pump_engaged` is now `AmplifierPump.pump_on`
        - `AmplifierPump.alc_engaged` is now `AmplifierPump.alc_on`
        - `AmplifierPump.use_probe` is now `AmplifierPump.probe_on`
    """

    #: Unique identifier. If left blank, a new unique ID will be generated.
    uid: str = field(default_factory=amplifier_pump_id_generator)

    pump_frequency: float | Parameter | None = None
    pump_power: float | Parameter | None = None
    pump_on: bool = True
    pump_filter_on: bool = True
    cancellation_on: bool = True
    cancellation_phase: float | Parameter | None = None
    cancellation_attenuation: float | Parameter | None = None
    cancellation_source: CancellationSource = CancellationSource.INTERNAL
    cancellation_source_frequency: float | None = None
    alc_on: bool = True
    probe_on: bool = False
    probe_frequency: float | Parameter | None = None
    probe_power: float | Parameter | None = None
