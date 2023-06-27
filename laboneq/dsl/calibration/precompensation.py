# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.types.enums import HighPassCompensationClearing
from laboneq.dsl.calibration.observable import Observable, RecursiveObservable
from laboneq.dsl.dsl_dataclass_decorator import classformatter

precompensation_id = 0


def precompensation_id_generator():
    global precompensation_id
    retval = f"mc{precompensation_id}"
    precompensation_id += 1
    return retval


@classformatter
@dataclass
class ExponentialCompensation(Observable):
    """Data object containing exponential filter parameters for the signal precompensation"""

    #: Exponential filter timeconstant
    timeconstant: float = 1e-6
    #: Exponential filter amplitude
    amplitude: float = 0.0


@classformatter
@dataclass
class HighPassCompensation(Observable):
    """Data object containing highpass filter parameters for the signal precompensation.

    .. versionchanged:: 2.8

        Deprecated `clearing` argument: It has no functionality.
    """

    #: high-pass filter time constant
    timeconstant: float = 1e-6
    #: Deprecated. Choose the clearing mode of the high-pass filter
    clearing: HighPassCompensationClearing = field(default=None)

    def __post_init__(self):
        if self.clearing is not None:
            warnings.warn(
                "`HighPassCompensation` argument `clearing` will be removed in the future versions. It has no functionality.",
                FutureWarning,
            )
        else:
            self.clearing = HighPassCompensationClearing.RISE
        super().__post_init__()


@classformatter
@dataclass
class FIRCompensation(Observable):
    """Data object containing FIR filter parameters for the signal precompensation"""

    #: FIR filter coefficients
    coefficients: ArrayLike = field(default_factory=lambda: np.zeros(40))


@classformatter
@dataclass
class BounceCompensation(Observable):
    """Data object containing parameters for the bounce compensation component of the signal precompensation"""

    #: Delay time to compensate
    delay: float = 0.0
    #: bounce compensation amplitude
    amplitude: float = 0.0


@classformatter
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
