# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

precompensation_id = 0


def precompensation_id_generator():
    global precompensation_id
    retval = f"pc{precompensation_id}"
    precompensation_id += 1
    return retval


@classformatter
@attrs.define
class ExponentialCompensation:
    """Parameters for exponential under- and overshoot compensation.

    Used to compensate for distortions due to LCR elements
    such as inductors, resistors, and capacitors.

    See [Filter Chain Specification](https://docs.zhinst.com/hdawg_user_manual/functional_description/pre_compensation.html#_filter_chain_specification)
    for a description of the filter.

    Attributes:
        timeconstant (float):
            Exponential filter time constant. Default: `1e-6`.
        amplitude (float):
            Exponential filter amplitude. Default: `0.0`.

    !!! note
        Only supported on the HDAWG with the
        [precompensation option](https://www.zhinst.com/ch/en/products/hdawg-pc-real-time-precompensation).
        Ignored on other devices.
    """

    # Exponential filter timeconstant
    timeconstant: float = 1e-6
    # Exponential filter amplitude
    amplitude: float = 0.0


@classformatter
@attrs.define
class HighPassCompensation:
    """Parameters for highpass filter signal precompensation.

    Used to compensate for distortions due to AC-coupling, DC-blocks and
    Bias-tees.

    See [Filter Chain Specification](https://docs.zhinst.com/hdawg_user_manual/functional_description/pre_compensation.html#_filter_chain_specification)
    for a description of the filter.

    Attributes:
        timeconstant (float):
            High-pass filter time constant. Default: `1e-6`.

    !!! note
        Only supported on the HDAWG with the
        [precompensation option](https://www.zhinst.com/ch/en/products/hdawg-pc-real-time-precompensation).
        Ignored on other devices.

    !!! version-removed "Removed in version 2.57.0"
        Removed the `.clearing` attribute that was deprecated in version 2.8.0.
        It had no effect.
    """

    # high-pass filter time constant
    timeconstant: float = 1e-6


@classformatter
@attrs.define
class FIRCompensation:
    """Parameters for FIR filter signal precompensation.

    Used to compensate for short time-scale distortions.

    The FIR filter performs a convolution of the input signal with
    the kernel specified by the coefficients below.

    See [Filter Chain Specification](https://docs.zhinst.com/hdawg_user_manual/functional_description/pre_compensation.html#_filter_chain_specification)
    for a description of the filter.

    Attributes:
        coefficients:
            Coefficients for the FIR filter convolution kernel.
            The first 8 coefficients are directly applied to the first eight
            taps of the FIR filter at the full time resolution. The remaining
            32 coefficients are applied to pairs of taps.
            Default: `np.zeros(40)`.

    !!! note
        Only supported on the HDAWG with the
        [precompensation option](https://www.zhinst.com/ch/en/products/hdawg-pc-real-time-precompensation).
        Ignored on other devices.
    """

    # FIR filter coefficients
    coefficients: ArrayLike = attrs.field(factory=lambda: np.zeros(40))

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FIRCompensation):
            return np.allclose(self.coefficients, other.coefficients)
        else:
            return NotImplemented


@classformatter
@attrs.define
class BounceCompensation:
    """Parameters for the bounce correction component of the signal
    precompensation.

    Use to correct reflections resulting from impedance mismatches.

    Bounce compensation adds to the original input signal the input
    signal multiplied by a given amplitude and delayed by a given
    time.

    See [Filter Chain Specification](https://docs.zhinst.com/hdawg_user_manual/functional_description/pre_compensation.html#_filter_chain_specification)
    for a description of the filter.

    Attributes:
        delay:
            The time to delay the added copy of the input signal by.
            Default: `0.0`.
        amplitude:
            The factor to multiply the amplitude of the added copy
            of the input signal by.
            Default: `0.0`.

    !!! note
        Only supported on the HDAWG with the
        [precompensation option](https://www.zhinst.com/ch/en/products/hdawg-pc-real-time-precompensation).
        Ignored on other devices.
    """

    # Delay time to compensate
    delay: float = 0.0
    # bounce compensation amplitude
    amplitude: float = 0.0


@classformatter
@attrs.define
class Precompensation:
    """Signal precompensation parameters.

    Attributes:
        uid:
            Unique identifier. If left blank, a new unique ID will be generated.
        exponential:
            List of exponential precompensation filters. Default: `None`.
        high_pass:
            A high pass precompenstation filter. Default: `None`.
        bounce:
            A bounce precompensation filter. Default: `None`.
        FIR:
            A FIR precompensation filter. Default: `None`.

    Setting or leaving an attribute above to `None` will disable the
    corresponding filter.

    !!! note
        Setting a precompensation filter may introduce a signal delay.
        The LabOne Q compiler will take this delay into account, but
        it will still be visible in the pulse sheet and when measuring
        the signal by other means (e.g. using an oscilloscope).

        See [Filter Chain Specification](https://docs.zhinst.com/hdawg_user_manual/functional_description/pre_compensation.html#_filter_chain_specification)
        for details on the delays.

    !!! note
        Only supported on the HDAWG with the
        [precompensation option](https://www.zhinst.com/ch/en/products/hdawg-pc-real-time-precompensation).
        Ignored on other devices.
    """

    # Unique identifier. If left blank, a new unique ID will be generated.
    uid: str = attrs.field(factory=precompensation_id_generator)

    # Exponential precompensation filter
    exponential: Optional[List[ExponentialCompensation]] = attrs.field(default=None)

    # High-pass compensation
    high_pass: Optional[HighPassCompensation] = attrs.field(default=None)

    # Bounce compensation
    bounce: Optional[BounceCompensation] = attrs.field(default=None)

    # FIR filter coefficients
    FIR: Optional[FIRCompensation] = attrs.field(default=None)

    def is_nonzero(self) -> bool:
        """Returns True if any filters are set.

        Returns:
            True if any filters are defined. False otherwise.
        """
        return self.exponential or self.high_pass or self.bounce or self.FIR
