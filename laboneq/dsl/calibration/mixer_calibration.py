# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import List, Optional
from dataclasses import dataclass, field

from laboneq.dsl.calibration.observable import Observable

mixer_calib_id = 0


def mixer_calib_id_generator():
    global mixer_calib_id
    retval = f"mc{mixer_calib_id}"
    mixer_calib_id += 1
    return retval


@dataclass(init=True, repr=True, order=True)
class MixerCalibration(Observable):
    """Data object containing mixer calibration."""

    #: Unique identifier. If left blank, a new unique ID will be generated.
    uid: str = field(default_factory=mixer_calib_id_generator)

    #: DC voltage offsets, two vlaues (for I and Q channels), epxresses in volts.
    voltage_offsets: Optional[List[float]] = field(default=None)

    #: Matrix for correcting gain and phase mismatch between I and Q.
    correction_matrix: Optional[List[List[float]]] = field(default=None)
