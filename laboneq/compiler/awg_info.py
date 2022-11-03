# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.compiler import DeviceType
    from laboneq.compiler.awg_signal_type import AWGSignalType
    from laboneq.compiler.code_generator import SignalObj


@dataclass
class AWGInfo:
    device_id: str
    signal_type: AWGSignalType
    awg_number: int
    seqc: str
    device_type: DeviceType
    signal_channels: List[Tuple[str, int]] = field(default_factory=list)
    signals: List[SignalObj] = field(default_factory=list)
