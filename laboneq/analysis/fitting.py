# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Fitting functions for modeling results in common quantum computing experiments.
"""

from __future__ import annotations

import functools
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from numpy.typing import ArrayLike


def fit(
    func: Callable,
    x: ArrayLike,
    y: ArrayLike,
    *args: tuple[float, ...],
    bounds: tuple[list, list] | None = None,
    plot: bool = False,
) -> tuple[ArrayLike, ArrayLike]:
    """Fit the given model.

    This function is a lightweight wrapper around
    [scipy.optimize.curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).

    Arguments:
        func:
            The model to fit.
            The model function must accept an initial parameter `x`,
            the points to evaluate the function at,
            and return `y`, the values at those points.
            The function must also accept, as positional parameters,
            the parameters passed in `*args` to `fit`.
        x:
            The points to fit the function at.
        y:
            The data to fit.
        *args:
            The initial values of the model parameters.
            Only the parameters supplied here are fitted.
            Any parameters of `func` not passed here are not fitted
            and retain their default values.
        bounds:
            If specified, a tuple containing the `(lower, upper)` bounds
            for the model parameters. The `lower` and `upper` bounds
            are both lists of the same length as `args`. I.e. there
            should be a bound for each of the specified model parameters.
        plot:
            If True, also plot the fit using `matplotlib.pyplot`.

    Returns:
        popt:
            The fitted values of the model parameters.
        pcov:
            The covariance matrix of the fit. The standard deviations
            of the fitted values may be calculate using:
            `perr = np.sqrt(np.diag(pcov))`.

    Examples:
        ``` py
        def line(x, m, c):
            return m * x + c

        x = np.linspace(0, 10, 100)
        data = np.random(*x.shape)
        popt, pcov = fit(line, x, data, 1, 0, bounds=([0, 0], [10, 10]))
        ```
    """
    kw = {}
    if bounds is not None:
        kw["bounds"] = bounds
    popt, pcov = opt.curve_fit(func, x, y, p0=args, **kw)

    if plot:
        plt.plot(x, y, ".k")
        plt.plot(x, func(x, *popt), "-r")

    return popt, pcov


def _fitting_function(f):
    """Add a .fit fitting function attribute to the decorated function."""

    @functools.wraps(fit)
    def _fit(*args, **kw):
        return fit(f, *args, **kw)

    f.fit = _fit
    return f


@_fitting_function
def oscillatory(
    x: ArrayLike,
    frequency: float,
    phase: float,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> ArrayLike:
    """A function for modelling oscillartions such as Rabi oscillations.

    The form of the function is a cosine:

    $$
        f(x) = amplitude \\times \\cos(frequency \\times x + phase) + offset
    $$

    Calling this function evaluates it. One may also fit this function
    by calling `oscillatory.fit` which calls
    [fit][laboneq.analysis.fitting.fit] with this function.

    Arguments:
        x:
            An array of values to evaluate the function at.
        frequency:
            The frequency of the cosine.
        phase:
            The phase of the cosine.
        amplitude:
            The amplitude of the cosine.
        offset:
            The offset of the cosine.

    Returns:
        values:
            The values of the oscillatory function at the times `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(0, 10, 100)
        values = oscillatory(x, 2, np.pi / 2, 0.5, 0.1)
        ```
        Fit the function:
        ``` py
        x = np.linspace(0, 10, 100)
        popt, pcov = oscillatory.fit(x, values, 1, 0, 0, 0)
        frequency, phase, amplitude, offset = popt
        ```
    """
    return amplitude * np.cos(frequency * x + phase) + offset


@_fitting_function
def oscillatory_decay(
    x: ArrayLike,
    frequency: float,
    phase: float,
    decay_rate: float,
    amplitude: float = 1.0,
    offset: float = 0.0,
) -> ArrayLike:
    """A function for modelling decaying oscillations such as Ramsey
    decay.

    The form of the function is a decaying cosine:

    $$
        f(x) = amplitude \\times \\cos(frequency \\times x + phase)
               \\exp(-decay \\text{\\textunderscore} rate \\times x) + offset
    $$

    Calling this function evaluates it. One may also fit this function
    by calling `oscillatory_decay.fit` which calls
    [fit][laboneq.analysis.fitting.fit] with this function.

    Arguments:
        x:
            An array of values to evaluate the function at.
        frequency:
            The frequency of the cosine.
        phase:
            The phase of the cosine.
        decay_rate:
            The exponential decay rate.
        amplitude:
            The amplitude of the cosine.
        offset:
            The offset of the function.

    Returns:
        values:
            The values of the decaying oscillation function at the times `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(0, 10, 100)
        values = oscillatory_decay(x, 2, np.pi / 2, 0.1, 0.5, 0.1)
        ```
        Fit the function:
        ``` py
        x = np.linspace(0, 10, 100)
        popt, pcov = oscillatory_decay.fit(x, values, 1, 0, 0, 0, 0)
        frequency, phase, decay_rate, amplitude, offset = popt
        ```
    """
    return amplitude * np.cos(frequency * x + phase) * np.exp(-decay_rate * x) + offset


@_fitting_function
def exponential_decay(
    x: ArrayLike, decay_rate: float, offset: float, amplitude: float = 1.0
) -> ArrayLike:
    """A function for modelling exponential decay such as T1 or T2 decay.

    The form of the function is a decaying exponential:

    $$
        f(x) = amplitude \\times \\exp(-decay \\text{\\textunderscore} rate \\times x)
               + offset
    $$

    Calling this function evaluates it. One may also fit this function
    by calling `exponential_decay.fit` which calls
    [fit][laboneq.analysis.fitting.fit] with this function.

    Arguments:
        x:
            An array of values to evaluate the function at.
        decay_rate:
            The exponential decay rate.
        offset:
            The offset of the function.
        amplitude:
            The amplitude multiplying the exponential.

    Returns:
        values:
            The values of the decay function at the times `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(0, 10, 100)
        values = exponential_decay(x, 0.1, 0.5, 2.0)
        ```
        Fit the function:
        ``` py
        x = np.linspace(0, 10, 100)
        popt, pcov = exponential_decay.fit(x, values, 0, 0, 1.0)
        decay_rate, offset, amplitude = popt
        ```
    """
    return amplitude * np.exp(-decay_rate * x) + offset


@_fitting_function
def lorentzian(
    x: ArrayLike, width: float, position: float, amplitude: float, offset: float
) -> ArrayLike:
    """A function for modelling a Lorentzian spectrum.

    The form of the spectrum function is:

    $$
        f(x) = offset + amplitude \\times \\frac{width}{width^2 + (x - position)^2}
    $$

    An inverted spectrum may be modelled by specifying a negative amplitude.

    Calling this function evaluates it. One may also fit this function
    by calling `lorentzian.fit` which calls
    [fit][laboneq.analysis.fitting.fit] with this function.

    Arguments:
        x:
            An array of values to evaluate the function at.
        width:
            The width of the spectrum.
        position:
            The position of the spectrum peak.
        amplitude:
            The amplitude of the spectrum. Specify a negative amplitude
            for an inverted spectrum.
        offset:
            The offset of the spectrum.

    Returns:
        values:
            The values of the spectrum at the times `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(0, 10, 100)
        values = lorentzian(x, 2.0, 0.5, 3.0, 0.1)
        ```
        Fit the function:
        ``` py
        x = np.linspace(0, 10, 100)
        popt, pcov = lorentzian.fit(x, values, 1.0, 0.0, 1.0, 0.0)
        width, position, amplitude, offset = popt
        ```
    """
    return offset + amplitude * width / (width**2 + (x - position) ** 2)


@_fitting_function
def fano_lineshape(
    x: ArrayLike,
    width: float,
    position: float,
    amplitude: float,
    fano: float = 0.0,
    offset: float = 0.5,
) -> ArrayLike:
    """A function for modelling a Fano resonance.

    The form of the Fano line-shape function is:

    $$
        f(x) = offset + amplitude \\times
               \\frac{fano \\times width + x - position) ** 2}{
               width^2 + (x - position)^2}
    $$

    Calling this function evaluates it. One may also fit this function
    by calling `fano_lineshape.fit` which calls
    [fit][laboneq.analysis.fitting.fit] with this function.

    Arguments:
        x:
            An array of values to evaluate the function at.
        width:
            The width of the resonance.
        position:
            The position of the resonance peak.
        amplitude:
            The amplitude of the resonance.
        fano:
            The Fano parameter.
        offset:
            The offset of the resonance.

    Returns:
        values:
            The values of the line-shape at the times `x`.

    Examples:
        Evaluate the function:
        ``` py
        x = np.linspace(0, 10, 100)
        values = fano_lineshape(x, 2.0, 0.5, 3.0, 1.0, 0.5, 0.1)
        ```
        Fit the function:
        ``` py
        x = np.linspace(0, 10, 100)
        popt, pcov = fano_lineshape.fit(x, values, 1.0, 0.0, 1.0, 0.0, 0.0)
        width, position, amplitude = popt
        ```
    """
    return offset + amplitude * (fano * width + x - position) ** 2 / (
        width**2 + (x - position) ** 2
    )
