# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import NDArray
import numpy as np
from zhinst.utils.shfqa.multistate import QuditSettings
from laboneq.dsl.experiment import pulse_library as pl


def calculate_integration_kernels(
    state_traces: list[NDArray],
) -> list[pl.PulseSampledComplex]:
    """Calculates the optimal kernel arrays for state discrimination given a set of
    reference traces corresponding to the states. The calculated kernels can directly be
    used as kernels in acquire statements.

    Args:
        state_traces: List of complex-valued reference traces, one array per state. The
            reference traces are typically obtained by an averaged scope measurement of
            the readout resonator response when the qudit is prepared in a certain
            state.

    Raises:
        ValueError: If any element of `state_traces` contains NaN values.

    """
    if any(np.any(np.isnan(trace)) for trace in state_traces):
        raise ValueError("`state_traces` contain NaN values.")

    n_traces = len(state_traces)
    settings = QuditSettings(state_traces)
    weights = settings.weights[: n_traces - 1]
    return [pl.sampled_pulse_complex(weight.vector) for weight in weights]
