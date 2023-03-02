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
    """Data object containing exponential filter parameters for the signal precompensation"""

    #: Exponential filter timeconstant
    timeconstant: float = 1e-6
    #: Exponential filter amplitude
    amplitude: float = 0.0


@dataclass
class HighPassCompensation(Observable):
    """Data object containing highpass filter parameters for the signal precompensation"""

    #: high-pass filter time constant
    timeconstant: float = 1e-6
    #: choose the clearing mode of the high-pass filter
    clearing: HighPassCompensationClearing = HighPassCompensationClearing.RISE


@dataclass
class FIRCompensation(Observable):
    """Data object containing FIR filter parameters for the signal precompensation"""

    #: FIR filter coefficients
    coefficients: ArrayLike = field(default_factory=lambda: np.zeros(40))


@dataclass
class BounceCompensation(Observable):
    """Data object containing parameters for the bounce compensation component of the signal precompensation"""

    #: Delay time to compensate
    delay: float = 0.0
    #: bounce compensation amplitude
    amplitude: float = 0.0


@dataclass(init=True, repr=True, order=True)
class Precompensation(RecursiveObservable):
    """Data object containing a collection of parameters for the different filters possible to enable for precompensation of signal distortion."""

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
