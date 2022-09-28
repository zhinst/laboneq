# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from dataclasses import dataclass, field

from numpy.typing import ArrayLike


@dataclass(init=True, repr=True, order=True)
class Waveform:
    data: ArrayLike = field(default=None)
    sampling_frequency: float = field(default=None)
    time_axis: ArrayLike = field(default=None)
    time_axis_at_port: ArrayLike = field(default=None)
    uid: str = field(default=None)
