# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OscillatorInfo:
    id: str
    frequency: float
    frequency_param: str
    hardware: bool
    device_id: str = None
