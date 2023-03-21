# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DeviceInfo:
    id: str
    device_type: str
    serial: str
    server: str
    interface: str
    reference_clock_source: str
