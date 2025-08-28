# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass
from typing_extensions import TypeAlias


@dataclass
class SignalDelay:
    on_device: float


SignalDelays: TypeAlias = dict[str, SignalDelay]
