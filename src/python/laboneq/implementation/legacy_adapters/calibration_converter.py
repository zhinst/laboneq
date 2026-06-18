# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

from laboneq.core.path import remove_logical_signal_prefix
from laboneq.core.types.enums import ModulationType
from laboneq.core.types.units import Quantity
from laboneq.data import calibration

if TYPE_CHECKING:
    from laboneq.dsl import calibration as dsl_calibration


def convert_signal_calibration(
    signal_calibration: dsl_calibration.SignalCalibration | None,
) -> calibration.SignalCalibration:
    if signal_calibration is None:
        return calibration.SignalCalibration()
    new = calibration.SignalCalibration()
    new.oscillator = signal_calibration.oscillator
    if signal_calibration.local_oscillator is not None:
        if (
            signal_calibration.local_oscillator.modulation_type
            == ModulationType.SOFTWARE
        ):
            raise ValueError(
                "Encountered `ModulationType.SOFTWARE` in local oscillator configuration "
                "which is not allowed. Make sure modulation type for "
                "all local oscillator calibration settings is set to "
                "either `ModulationType.HARDWARE` or `ModulationType.AUTO`."
            )
        new.local_oscillator_frequency = signal_calibration.local_oscillator.frequency
    new.mixer_calibration = signal_calibration.mixer_calibration
    new.precompensation = signal_calibration.precompensation
    new.port_delay = signal_calibration.port_delay
    new.delay_signal = signal_calibration.delay_signal
    new.port_mode = signal_calibration.port_mode
    new.voltage_offset = signal_calibration.voltage_offset
    if signal_calibration.range is not None and not isinstance(
        signal_calibration.range, Quantity
    ):
        new.range = Quantity(value=signal_calibration.range, unit=None)
    else:
        new.range = signal_calibration.range
    new.threshold = signal_calibration.threshold
    new.amplitude = signal_calibration.amplitude
    new.amplifier_pump = signal_calibration.amplifier_pump
    for route in signal_calibration.added_outputs or []:
        route = copy.copy(route)
        route.source = remove_logical_signal_prefix(route.source)
        new.added_outputs.append(route)
    new.automute = signal_calibration.automute
    return new
