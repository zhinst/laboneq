# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Marker:
    marker_selector: str
    enable: bool
    start: float
    length: float
    pulse_id: str
