# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

def sample_pulse(
    *,
    signal_type: str,
    sampling_rate: float,
    length: float,
    amplitude: complex,
    pulse_function: str | None,
    modulation_frequency: float | None = None,
    phase: float | None = None,
    samples: ArrayLike | None = None,
    pulse_parameters: dict[str, Any] | None = None,
    subslice: tuple[int, int] | None = None,
) -> np.ndarray:
    """Sample a waveform from a pulse definition.

    Args:
        signal_type: "iq" if the pulse represents quadrature (IQ) modulation.
        sampling_rate: Sampling rate.
        length: Pulse length in seconds.
        amplitude: Amplitude for the pulse envelope function.
        pulse_function: In case of a functional pulse, the sampling function.
        modulation_frequency: Carrier frequency in Hz for software modulation. Default `None`.
        phase: Phase shift in radians to apply to the signal.
        samples: Pulse envelope samples for a sampled pulse.
        pulse_parameters: Additional pulse parameters, passed to the sampler.
        subslice: (start, end) sample indices of the pulse slice to be sampled.
            Defaults to the whole pulse.

    Returns:
        Sampled, scaled and possibly modulated waveform samples as a single `ndarray`.
            For `signal_type == "iq"` the array is complex-valued. For every other `signal_type`
            the imaginary part carries no information, and a real-valued array is returned.

    Raises:
        ValueError: Invalid combination of arguments, a non-numeric `amplitude`
            pulse parameter, or a pulse that fails to sample.
    """
