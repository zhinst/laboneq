# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from numpy.typing import NDArray
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

    """

    n_traces = len(state_traces)
    settings = QuditSettings(state_traces)
    weights = settings.weights[: n_traces - 1]
    return [pl.sampled_pulse_complex(weight.vector) for weight in weights]
