# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from laboneq.compiler.common.trigger_mode import TriggerMode

if TYPE_CHECKING:
    from laboneq.compiler.common import DeviceType
    from laboneq.compiler.common.awg_signal_type import AWGSignalType
    from laboneq.compiler.common.signal_obj import SignalObj


@dataclass
class AWGInfo:
    device_id: str
    signal_type: AWGSignalType
    awg_number: int
    seqc: str
    device_type: DeviceType
    sampling_rate: float
    trigger_mode: TriggerMode = TriggerMode.NONE
    reference_clock_source: Optional[str] = None
    signal_channels: List[Tuple[str, int]] = field(default_factory=list)
    signals: List[SignalObj] = field(default_factory=list)
