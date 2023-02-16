# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict

import numpy as np

from laboneq.core.utilities.pulse_sampler import pulse_function_library
from laboneq.dsl.experiment.pulse import (
    PulseFunctional,
    PulseSampledComplex,
    PulseSampledReal,
)


def register_pulse_functional(sampler: Callable, name: str = None):
    """Build & register a new pulse type from a sampler function.

    The sampler function must have the following signature:

    .. code-block:: python

        def sampler(x: ndarray, **pulse_params: Dict[str, Any]) -> ndarray:
            pass


    The vector ``x`` marks the points where the pulse function is to be evaluated. The
    values of ``x`` range from -1 to +1. The argument ``pulse_params`` contains all
    the sweep parameters, evaluated for the current iteration.
    In addition, ``pulse_params``  also contains the following keys:

    - ``length``: the true length of the pulse
    - ``amplitude``: the true amplitude of the pulse
    - ``sampling_rate``: the sampling rate

    Typically, the sampler function should discard ``length`` and ``amplitude``, and
    instead assume that the pulse extends from -1 to 1, and that it has unit
    amplitude. LabOne Q will automatically rescale the sampler's output to the correct
    amplitude and length.


    Args
        - sampler (Callable): the function used for sampling the pulse

        - name (str): the name used internally for referring to this pulse type

    Returns
        A factory function for new :py:class:`~.Pulse` objects. The return value has the
        following signature:

        .. code-block:: python

            def <name>(
                uid: str = None,
                length: float = 100e-9,
                amplitude: float = 1.0,
                **pulse_parameters: Dict[str, Any],
            ):
                pass


    """
    if name is None:
        function_name = sampler.__name__
    else:
        function_name = name

    def factory(
        uid: str = None,
        length: float = 100e-9,
        amplitude: float = 1.0,
        **pulse_parameters: Dict[str, Any],
    ):
        if pulse_parameters == {}:
            pulse_parameters = None
        if uid is None:
            return PulseFunctional(
                function=function_name,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
            )
        else:
            return PulseFunctional(
                function=function_name,
                uid=uid,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
            )

    factory.__name__ = function_name
    factory.__doc__ = sampler.__doc__
    # we do not wrap __qualname__, it throws off the documentation generator

    pulse_function_library[function_name] = sampler
    return factory


@register_pulse_functional
def gaussian(x, sigma=1 / 3, zero_boundaries=False, **_):
    """Create a gaussian pulse.

    Args:
        uid (str): Unique identifier of the pulse.
        length (float): Length of the pulse in seconds
        amplitude (float): Amplitude of the pulse
        sigma (float): Std. deviation, relative to pulse length
        zero_boundaries (bool): Whether to zero the pulse at the boundaries

    Returns:
        Gaussian pulse.
    """
    gauss = np.exp(-(x**2) / (2 * sigma**2))
    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        delta = np.exp(-(dt**2) / (2 * sigma**2))
        gauss -= delta
        gauss /= 1 - delta
    return gauss


@register_pulse_functional
def gaussian_square(
    x, sigma=1 / 3, width=90e-9, zero_boundaries=False, length=100e-9, **_
):
    """Create a gaussian square waveform with a square portion of length
    ``width`` and Gaussian shaped sides.

    Args:
        uid (str): Unique identifier of the pulse.
        length (float): Length of the pulse in seconds
        width (float): Width of the flat portion of the pulse in seconds
        amplitude (float): Amplitude of the pulse
        sigma (float): Std. deviation of the Gaussian rise/fall portion of the pulse
        zero_boundaries (bool): Whether to zero the pulse at the boundaries

    Returns:
        Gaussian square pulse.
    """

    risefall_in_samples = round(len(x) * (1 - width / length) / 2)
    flat_in_samples = len(x) - 2 * risefall_in_samples
    gauss_x = np.linspace(-1.0, 1.0, 2 * risefall_in_samples)
    gauss_part = np.exp(-(gauss_x**2) / (2 * sigma**2))
    gauss_sq = np.concatenate(
        (
            gauss_part[:risefall_in_samples],
            np.ones(flat_in_samples),
            gauss_part[risefall_in_samples:],
        )
    )
    if zero_boundaries:
        t_left = gauss_x[0] - (gauss_x[1] - gauss_x[0])
        delta = np.exp(-(t_left**2) / (2 * sigma**2))
        gauss_sq -= delta
        gauss_sq /= 1 - delta
    return gauss_sq


@register_pulse_functional
def const(x, **_):
    """Create a const pulse

    Args:
        uid (str): Unique identifier of the pulse
        length (float): Length of the pulse in seconds
        amplitude (float): Amplitude of the pulse

    Returns:
        Constant pulse.
    """
    return np.ones_like(x)


@register_pulse_functional
def triangle(x, **_):
    """Create a triangle pulse

    Args:
        uid (str): Unique identifier of the pulse
        length (float): Length of the pulse in seconds
        amplitude (float): Amplitude of the pulse

    Returns:
        Triangle pulse.
    """
    return 1 - np.abs(x)


@register_pulse_functional
def sawtooth(x, **_):
    """Create a sawtooth pulse

    Args:
        uid (str): Unique identifier of the pulse
        length (float): Length of the pulse in seconds
        amplitude (float): Amplitude of the pulse

    Returns:
        Sawtooth pulse.
    """

    return 0.5 * (1 - x)


@register_pulse_functional
def drag(x, sigma=1 / 3, beta=1.0, zero_boundaries=False, **_):
    """Create a DRAG pulse

    Args:
        uid (str): Unique identifier of the pulse
        length (float): Length of the pulse in seconds.
        amplitude (float): Amplitude of the pulse
        sigma (float): Std. deviation, relative to pulse length
        beta (float): Relative amplitude of the quadrature component
        zero_boundaries (bool): Whether to zero the pulse at the boundaries

    Returns:
        DRAG pulse.
    """
    gauss = np.exp(-(x**2) / (2 * sigma**2))
    delta = 0
    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        delta = np.exp(-(dt**2) / (2 * sigma**2))
    d_gauss = -x / sigma**2 * gauss
    gauss -= delta
    return (gauss + 1j * beta * d_gauss) / (1 - delta)


@register_pulse_functional
def cos2(x, **_):
    """Create a raised cosine pulse

    Args:
        uid (str): Unique identifier of the pulse
        length (float): Length of the pulse in seconds.
        amplitude (float): Amplitude of the pulse

    Returns:
        Raised cosine pulse.
    """
    return np.cos(x * np.pi / 2) ** 2


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
