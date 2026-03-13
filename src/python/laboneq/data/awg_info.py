# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from laboneq.core.types.enums.trigger_mode import TriggerMode

if TYPE_CHECKING:
    from laboneq.compiler.common.device_type import DeviceType
    from laboneq.compiler.common.signal_obj import SignalObj
    from laboneq.core.types.enums.awg_signal_type import AWGSignalType


@dataclass(init=True, repr=True, order=True, frozen=True)
class AwgKey:
    device_id: str
    awg_id: int | str


@dataclass
class AWGInfo:
    device_id: str
    awg_id: int | str  # actual type depends on device class
    device_type: DeviceType
    device_class: int = 0x0
    trigger_mode: TriggerMode = TriggerMode.NONE
    signal_type: AWGSignalType | None = None
    awg_allocation: list[int] = field(default_factory=list)
    signal_channels: list[tuple[str, int]] = field(default_factory=list)
    signals: list[SignalObj] = field(default_factory=list)

    @property
    def key(self) -> AwgKey:
        return AwgKey(self.device_id, self.awg_id)
