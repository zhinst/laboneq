# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Tuple

from laboneq.compiler.common.trigger_mode import TriggerMode

if TYPE_CHECKING:
    from laboneq.compiler.common.awg_signal_type import AWGSignalType
    from laboneq.compiler.common.device_type import DeviceType
    from laboneq.compiler.common.signal_obj import SignalObj


@dataclass(init=True, repr=True, order=True, frozen=True)
class AwgKey:
    device_id: str
    awg_number: int


@dataclass
class AWGInfo:
    device_id: str
    signal_type: AWGSignalType
    awg_number: int
    device_type: DeviceType
    sampling_rate: float
    device_class: int = 0x0
    trigger_mode: TriggerMode = TriggerMode.NONE
    reference_clock_source: str | None = None
    signal_channels: List[Tuple[str, int]] = field(default_factory=list)
    signals: List[SignalObj] = field(default_factory=list)

    @property
    def key(self) -> AwgKey:
        return AwgKey(self.device_id, self.awg_number)
