# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from laboneq.compiler.common.trigger_mode import TriggerMode

if TYPE_CHECKING:
    from laboneq.compiler.common.awg_signal_type import AWGSignalType
    from laboneq.compiler.common.device_type import DeviceType
    from laboneq.compiler.common.signal_obj import SignalObj


@dataclass(init=True, repr=True, order=True, frozen=True)
class AwgKey:
    device_id: str
    awg_id: int | str


@dataclass
class AWGInfo:
    device_id: str
    signal_type: AWGSignalType
    awg_id: int | str  # actual type depends on device class
    device_type: DeviceType
    sampling_rate: float
    device_class: int = 0x0
    trigger_mode: TriggerMode = TriggerMode.NONE
    reference_clock_source: str | None = None
    signal_channels: list[tuple[str, int]] = field(default_factory=list)
    signals: list[SignalObj] = field(default_factory=list)
    oscs: dict[str, int] = field(default_factory=dict)

    @property
    def key(self) -> AwgKey:
        return AwgKey(self.device_id, self.awg_id)
