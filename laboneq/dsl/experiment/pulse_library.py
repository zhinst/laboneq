# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.dsl.enums import PulseType
from .pulse import PulseFunctional, PulseSampledReal, PulseSampledComplex


def gaussian_q(uid=None, function=PulseType.INTERNAL, length=100e-9, amplitude=1.0):
    """Create a Gaussian Q pulse.

    Args:
        uid (str): Unique identifier of the pulse.
        function (PulseType): Pulse type, on of "gaussian", "const" and "internal". Default is internal.
        length (float): Length of the pulse in seconds.
        amplitude (float): Amplitude of the pulse in Volt.

    Returns:
        Gaussian Q pulse.
    """
    if uid is None:
        return PulseFunctional(function=function, length=length, amplitude=amplitude)
    else:
        return PulseFunctional(
            uid=uid, function=function, length=length, amplitude=amplitude
        )


def gaussian(uid=None, function=PulseType.GAUSSIAN, length=100e-9, amplitude=1.0):
    """Create of a gaussian pulse.

    Args:
        uid (str): Unique identifier of the pulse.
        function (PulseType): Pulse type, on of "gaussian", "const" and "internal". Default is gaussian.
        length (float): Length of the pulse in seconds.
        amplitude (float): Amplitude of the pulse in Volt.

    Returns:
        Gaussian pulse.
    """
    if uid is None:
        return PulseFunctional(function=function, length=length, amplitude=amplitude)
    else:
        return PulseFunctional(
            uid=uid, function=function, length=length, amplitude=amplitude
        )


def const(uid=None, function=PulseType.CONST, length=100e-9, amplitude=1.0):
    """Creation of const pulse

    Args:
        uid (str): Unique identifier of the pulse
        function (PulseType): Pulse type, on of "gaussian", "const" and "internal". Default is const.
        length (float): Length of the pulse in seconds.
        amplitude (float): Amplitude of the pulse in Volt.

    Returns:
        Constant pulse.
    """
    if uid is None:
        return PulseFunctional(function=function, length=length, amplitude=amplitude)
    else:
        return PulseFunctional(
            uid=uid, function=function, length=length, amplitude=amplitude
        )


def sampled_pulse_real(samples, uid=None):
    """Create a pulse based on a array of real values.

    Args:
        samples: Real valued data.
        uid: Unique identifier of the created pulse.

    Returns:
        Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampledReal(samples=samples)
    else:
        return PulseSampledReal(uid=uid, samples=samples)


def sampled_pulse_complex(samples, uid=None):
    """Create a pulse based on a array of complex values.

    Args:
        samples: Complex valued data.
        uid: Unique identifier of the created pulse.

    Returns:
        Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampledComplex(samples=samples)
    else:
        return PulseSampledComplex(uid=uid, samples=samples)
