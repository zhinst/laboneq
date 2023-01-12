# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional

from laboneq.core.types.enums.mixer_type import MixerType

if TYPE_CHECKING:
    from laboneq.compiler.common.awg_info import AWGInfo
    from laboneq.compiler.common.device_type import DeviceType


@dataclass(init=True, repr=True, order=True)
class SignalObj:
    id: str
    sampling_rate: float
    start_delay: float
    delay_signal: float
    signal_type: str
    device_id: str
    device_type: DeviceType
    oscillator_frequency: float = None  # for software modulation only
    pulses: List = field(default_factory=list)
    channels: List = field(default_factory=list)
    awg: AWGInfo = None
    total_delay: float = None
    on_device_delay: float = 0
    mixer_type: Optional[MixerType] = None
    hw_oscillator: Optional[str] = None
