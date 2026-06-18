# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from laboneq.data import EnumReprMixin

if TYPE_CHECKING:
    from laboneq.core.types.enums import PortMode
    from laboneq.core.types.units import Quantity
    from laboneq.dsl.calibration import (
        AmplifierPump,
        MixerCalibration,
        Oscillator,
        OutputRoute,
        Precompensation,
    )
    from laboneq.dsl.parameter import Parameter


class CancellationSource(EnumReprMixin, Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"


@dataclass
class SignalCalibration:
    oscillator: Oscillator | None = None
    local_oscillator_frequency: float | Parameter | None = None
    mixer_calibration: MixerCalibration | None = None
    precompensation: Precompensation | None = None
    port_delay: float | Parameter | None = None
    port_mode: PortMode | None = None
    delay_signal: float | None = None
    voltage_offset: float | None = None
    range: Quantity | None = None
    threshold: float | list[float] | None = None
    amplitude: float | Parameter | None = None
    amplifier_pump: AmplifierPump | None = None
    added_outputs: list[OutputRoute] = field(default_factory=list)
    automute: bool = False
