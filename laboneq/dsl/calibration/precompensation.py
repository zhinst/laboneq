# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.types.enums import HighPassCompensationClearing
from laboneq.dsl.calibration.observable import Observable, RecursiveObservable

precompensation_id = 0


def precompensation_id_generator():
    global precompensation_id
    retval = f"mc{precompensation_id}"
    precompensation_id += 1
    return retval


@dataclass
class ExponentialCompensation(Observable):
    """Data object containing exponential compensation parameters"""

    timeconstant: float = 1e-6
    amplitude: float = 0.0


@dataclass
class HighPassCompensation(Observable):
    """Data object containing highpass compensation parameters"""

    timeconstant: float = 1e-6
    clearing: HighPassCompensationClearing = HighPassCompensationClearing.RISE


@dataclass
class FIRCompensation(Observable):
    """Data object containing FIR filter compensation parameters"""

    coefficients: ArrayLike = field(default_factory=lambda: np.zeros(40))


@dataclass
class BounceCompensation(Observable):
    """Data object containing bounce compensation parameters"""

    delay: float = 0.0
    amplitude: float = 0.0


@dataclass(init=True, repr=True, order=True)
class Precompensation(RecursiveObservable):
    """Data object containing mixer calibration."""

    #: Unique identifier. If left blank, a new unique ID will be generated.
    uid: str = field(default_factory=precompensation_id_generator)

    #: Exponential precompensation filter; pairs of time constant and amplitude.
    exponential: Optional[List[ExponentialCompensation]] = field(default=None)

    #: High-pass compensation.
    high_pass: Optional[HighPassCompensation] = field(default=None)

    #: Bounce compensation
    bounce: Optional[BounceCompensation] = field(default=None)

    #: FIR filter coefficients
    FIR: Optional[FIRCompensation] = field(default=None)

    def is_nonzero(self):
        return self.exponential or self.high_pass or self.bounce or self.FIR
