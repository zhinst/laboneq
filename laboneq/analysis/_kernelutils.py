# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import warnings

import numpy as np
from numpy.typing import NDArray
from zhinst.utils.shfqa.multistate import QuditSettings

from laboneq.dsl.experiment import pulse_library as pl


def calculate_integration_kernels(
    state_traces: list[NDArray],
) -> list[pl.PulseSampledComplex]:
    """Calculates the optimal kernel arrays for state discrimination given a set of
    reference traces corresponding to measurement of each qubit state.
    The calculated kernels can directly be used as kernels in acquire statements.

    Args:
        state_traces: List of complex-valued reference traces, one array per state. The
            reference traces are typically obtained by an averaged scope measurement of
            the readout resonator response when the qudit is prepared in a certain
            state.

    Raises:
        ValueError: If any element of `state_traces` contains NaN values.

    !!! version-changed "Deprecated in version 2.26.0"
        Deprecated in favor of `calculate_integration_kernels_thresholds`
        which additionally supplies the threshold information.
    """

    warnings.warn(
        "Deprecated in favor of `calculate_integration_kernels_thresholds` "
        "which additionally supplies the threshold information.",
        FutureWarning,
        stacklevel=2,
    )
    if any(np.any(np.isnan(trace)) for trace in state_traces):
        raise ValueError("`state_traces` contain NaN values.")

    n_traces = len(state_traces)
    settings = QuditSettings(state_traces)
    weights = settings.weights[: n_traces - 1]
    return [pl.sampled_pulse_complex(weight.vector) for weight in weights]


def calculate_integration_kernels_thresholds(
    state_traces: list[NDArray],
) -> tuple[list[pl.PulseSampledComplex], list[float]]:
    """Calculates the optimal kernel arrays and threshold values for state discrimination given a set of
    reference traces corresponding to measurement of each qubit state.

    Args:
        state_traces: List of complex-valued reference traces, one array per state. The
            reference traces are typically obtained by an averaged scope measurement of
            the readout resonator response when the qudit is prepared in a certain
            state.

    Returns:
        kernels: List of kernels to be used directly as an argument to `acquire` statements.
        thresholds: List of thresholds that can be used in the `threshold`
            setting when calibrating acquire signals.

    Raises:
        ValueError: If any element of `state_traces` contains NaN values.

    !!! version-changed "Added in version 2.26.0"
        Added extended functionality to additionally supply threshold information.
    """
    if any(np.any(np.isnan(trace)) for trace in state_traces):
        raise ValueError("`state_traces` contain NaN values.")

    n_traces = len(state_traces)
    settings = QuditSettings(state_traces)
    weights = settings.weights[: n_traces - 1]
    thresholds = settings.thresholds
    return [pl.sampled_pulse_complex(weight.vector) for weight in weights], thresholds
