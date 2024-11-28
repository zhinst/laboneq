# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np

from laboneq.core.utilities.pulse_sampler import _pulse_samplers, _pulse_factories
from laboneq.dsl.experiment.pulse import (
    PulseFunctional,
    PulseSampled,
    PulseSampledComplex,
    PulseSampledReal,
)

# deprecated alias for _pulse_samples, use pulse_library.pulse_sampler(...) instead:
pulse_function_library = _pulse_samplers


def register_pulse_functional(sampler: Callable, name: str | None = None):
    """Build & register a new pulse type from a sampler function.

    The sampler function must have the following signature:

    ``` py

        def sampler(x: ndarray, **pulse_params: Dict[str, Any]) -> ndarray:
            pass
    ```

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


    Args:
        sampler:
            the function used for sampling the pulse
        name:
            the name used internally for referring to this pulse type

    Returns:
        pulse_factory (function):
            A factory function for new ``Pulse`` objects.
            The return value has the following signature:
            ``` py

                def <name>(
                    uid: str = None,
                    length: float = 100e-9,
                    amplitude: float = 1.0,
                    **pulse_parameters: Dict[str, Any],
                ):
                    pass
            ```
    """
    if name is None:
        function_name = sampler.__name__
    else:
        function_name = name

    def factory(
        uid: str | None = None,
        length: float = 100e-9,
        amplitude: float = 1.0,
        can_compress=False,
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
                can_compress=can_compress,
            )
        else:
            return PulseFunctional(
                function=function_name,
                uid=uid,
                length=length,
                amplitude=amplitude,
                pulse_parameters=pulse_parameters,
                can_compress=can_compress,
            )

    factory.__name__ = function_name
    factory.__doc__ = sampler.__doc__
    # we do not wrap __qualname__, it throws off the documentation generator

    _pulse_samplers[function_name] = sampler
    _pulse_factories[function_name] = factory
    return factory


@register_pulse_functional
def gaussian(
    x,
    sigma=1 / 3,
    order=2,
    zero_boundaries=False,
    **_,
):
    """Create a Gaussian pulse.

    Returns a generalised Gaussian pulse with order parameter $n$, defined by:

    $$ g(x, \\sigma, n) = e^{-\\left(\\frac{x^2}{2\\sigma^2}\\right)^{\\frac{n}{2}}} $$.

    When the order $n = 2$, the formula simplifies to the standard Gaussian:

    $$ g(x, \\sigma_0) = e^{-\\frac{x^2}{2\\sigma_0^2}} $$

    For higher orders ($n > 2$), the value of $\\sigma$ is adjusted so that the
    pulse has the same near-zero values at the edges as the ordinary Gaussian.

    In general, for $x \\in [-L, L]$, the adjusted $\\sigma$ can be written as:

    $$\\sigma = \\frac{\\sigma_0^{\\frac{2}{n}}}{2^{\\left(\\frac{n-2}{2 n}\\right)} L^{\\left(\\frac{2-n}{n}\\right)}}$$

    Considering here $x \\in [-1, 1]$, the adjusted $\\sigma$ simplifies to:

    $$\\sigma = \\frac{\\sigma_0^{\\frac{2}{n}}}{2^{\\left(\\frac{n-2}{2 n}\\right)}}$$

    Arguments:
        sigma (float):
            Standard deviation relative to the interval the pulse is sampled from, here [-1, 1]. Defaults
                to 1/3.
        order (int):
            Order of the Gaussian pulse, must be positive and even, default is 2 (standard Gaussian), order > 2 will create a super Gaussian pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries, default is False

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse


    Returns:
        pulse (Pulse): Gaussian pulse.
    """
    if order < 2 or order % 2 != 0:
        raise ValueError("The order must be a positive and even integer.")
    elif order == 2:
        gauss = np.exp(-(x**2 / (2 * sigma**2)))
    elif order > 2:
        sigma_updated = (sigma ** (2 / order)) / (2 ** ((order - 2) / (2 * order)))
        gauss = np.exp(-((x**2 / (2 * sigma_updated**2)) ** (order / 2)))

    if zero_boundaries:
        dt = x[0] - (x[1] - x[0])
        dt = np.abs(dt)
        if order == 2:
            delta = np.exp(-(dt**2 / (2 * sigma**2)))
        else:
            sigma_updated = (sigma ** (2 / order)) / (
                2 ** ((order - 2) / (2 * order)) * dt ** ((2 - order) / order)
            )
            delta = np.exp(-((dt**2 / (2 * sigma_updated**2)) ** (order / 2)))
        gauss -= delta
        gauss /= 1 - delta
    return gauss


@register_pulse_functional
def gaussian_square(x, sigma=1 / 3, width=None, zero_boundaries=False, *, length, **_):
    """Create a gaussian square waveform with a square portion of length
    ``width`` and Gaussian shaped sides.

    Arguments:
        length (float):
            Length of the pulse in seconds
        width (float):
            Width of the flat portion of the pulse in seconds. Dynamically set to 90% of `length` if not provided.
        sigma (float):
            Std. deviation of the Gaussian rise/fall portion of the pulse
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Gaussian square pulse.
    """

    if width is not None and width >= length:
        raise ValueError(
            "The width of the flat portion of the pulse must be smaller than the total length."
        )

    if width is None:
        width = 0.9 * length

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
    """Create a constant pulse.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Constant pulse.
    """
    return np.ones_like(x)


@register_pulse_functional
def triangle(x, **_):
    """Create a triangle pulse.

    A triangle pulse varies linearly from a starting amplitude of
    zero, to a maximum amplitude of one in the middle of the pulse,
    and then back to a final amplitude of zero.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Triangle pulse.
    """
    return 1 - np.abs(x)


@register_pulse_functional
def sawtooth(x, **_):
    """Create a sawtooth pulse.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Sawtooth pulse.
    """

    return 0.5 * (1 - x)


@register_pulse_functional
def drag(x, sigma=1 / 3, beta=0.2, zero_boundaries=False, **_):
    """Create a DRAG pulse.

    Arguments:
        sigma (float):
            Standard deviation relative to the interval the pulse is sampled from, here [-1, 1]. Defaults
        beta (float):
            Relative amplitude of the quadrature component
        zero_boundaries (bool):
            Whether to zero the pulse at the boundaries

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): DRAG pulse.
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
    """Create a raised cosine pulse.

    Keyword Arguments:
        uid ([str][]): Unique identifier of the pulse
        length ([float][]): Length of the pulse in seconds
        amplitude ([float][]): Amplitude of the pulse

    Returns:
        pulse (Pulse): Raised cosine pulse.
    """
    return np.cos(x * np.pi / 2) ** 2


def sampled_pulse(samples, uid=None, can_compress=False):
    """Create a pulse based on a array of waveform values.

    Arguments:
        samples (numpy.ndarray): waveform envelope data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampled(samples=samples, can_compress=can_compress)
    else:
        return PulseSampled(uid=uid, samples=samples, can_compress=can_compress)


def sampled_pulse_real(samples, uid=None, can_compress=False):
    """Create a pulse based on a array of real values.

    Arguments:
        samples (numpy.ndarray): Real valued data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampledReal(samples=samples, can_compress=can_compress)
    else:
        return PulseSampledReal(uid=uid, samples=samples, can_compress=can_compress)


def sampled_pulse_complex(samples, uid=None, can_compress=False):
    """Create a pulse based on a array of complex values.

    Args:
        samples (numpy.ndarray): Complex valued data.
        uid (str): Unique identifier of the created pulse.

    Returns:
        pulse (Pulse): Pulse based on the provided sample values.
    """
    if uid is None:
        return PulseSampledComplex(samples=samples, can_compress=can_compress)
    else:
        return PulseSampledComplex(uid=uid, samples=samples, can_compress=can_compress)


def pulse_sampler(name: str) -> Callable:
    """Return the named pulse sampler.

    The sampler is the original function used to define the pulse.

    For example in:

        ```python
        @register_pulse_functional
        def const(x, **_):
            return numpy.ones_like(x)
        ```

    the sampler is the *undecorated* function `const`. Calling
    `pulse_sampler("const")` will return this undecorated function.

    This undecorate function is called a "sampler" because it is used by
    the LabOne Q compiler to generate the samples played by a pulse.

    Arguments:
        name: The name of the sampler to return.

    Return:
        The sampler function.
    """
    return _pulse_samplers[name]


def pulse_factory(name: str) -> Callable:
    """Return the named pules factory.

    The pulse factory returns the description of the pulse used to specify
    a pulse when calling LabOne Q DSl commands such as `.play(...)` and
    `.measure(...)`.

    For example, in:

        ```python
        @register_pulse_functional
        def const(x, **_):
            return numpy.ones_like(x)
        ```

    the factory is the *decorated* function `const`. Calling
    `pulse_factory("const")` will return this decorated function. This is
    the same function one calls when calling `pulse_library.const(...)`.

    Arguments:
        name: The name of the factory to return.

    Return:
        The factory function.
    """
    return _pulse_factories[name]
