# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SignalInfo:
    signal_id: str
    signal_type: str
    device_id: str
    device_serial: str
    device_type: str
    connection_type: str
    channels: str
    delay_signal: float
    modulation: str
    offset: float
