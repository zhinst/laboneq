# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Helper functions to shift waveform outputs with sub-sample precision"""

import numpy as np

from laboneq.dsl.calibration import FIRCompensation, Precompensation, SignalCalibration


def fractional_delay_filter(
    mu: float, fs_hz: float, f_bw_hz: float, N: int, beta: float = 6.0
) -> tuple[np.array, float]:
    """Create a sinc FIR filter with fractional delay and tunable bandwidth.

    Args:
        mu (float): fractional delay in samples (must be -0.5 <= mu <= 0.5)
        fs_hz (float): sampling rate in Hz
        f_bw_hz (float): cutoff frequency in Hz (must be 0 < f_bw_hz < fs_hz/2)
                         Keep f_bw_hz <= ~0.45*fs_hz for good in-band behavior.
        N (int): Length of the FIR filter
        beta (float, optional): Kaiser window beta (≈5–8 typical). Defaults to 6.0.

    Raises:
        ValueError: If one or more parameters are out of bounds

    Returns:
        fir (np.array): FIR filter array
        tau_g_samples (float): the group delay in samples
    """

    if not (0 < f_bw_hz < 0.5 * fs_hz):
        raise ValueError("f_bw_hz must be in (0, fs_hz/2).")
    if not (-0.5 <= mu <= 0.5):
        raise ValueError(
            "mu should be in [-0.5, 0.5] after separating the integer delay."
        )

    # To have the best result, add an additional delay
    # to make it fully causal and capture correctly the main lobe of the filter
    M = (N - 1) / 2.0

    n = np.arange(N)
    t = n - M - mu

    # Discrete-time cutoff factor
    x = 2.0 * f_bw_hz / fs_hz

    # Ideal lowpass kernel at cutoff f_bw_hz, shifted by mu (np.sinc uses sin(pi x)/(pi x))
    h = np.sinc(x * t)

    # Apply window
    w = np.kaiser(N, beta)
    h *= w

    # Normalize for unity DC gain
    h /= h.sum()

    # Group delay
    tau_g_samples = M + mu

    return h, tau_g_samples


def get_delay_settings(
    delay: float, fs_hz: float, f_bw_hz: float = 900e6, beta: float = 6.0
) -> tuple[float, np.array, float]:
    """Get the device parameters (FIR filter and port/node delay) to get the given delay
    on a HDAWG output.

    Args:
        delay (float): The desired delay in seconds
        fs_hz (float): sampling rate in Hz
        f_bw_hz (float, optional): cutoff frequency in Hz (must be 0 < f_bw_hz < fs_hz/2)
                                   Defaults to 900e6.
        beta (float, optional): Kaiser window beta (≈5–8 typical). Defaults to 6.0.

    Raises:
        ValueError: If one or more parameters are out of bounds

    Returns:
        port_delay (float): the port (node) delay in seconds
        fir (np.array): FIR filter array
        tau_g_samples (float): the group delay in seconds
    """

    # Minimum delay using the FIR filter
    min_delay = 3.5
    # The maximum delay is given by the digital delay (port delay)
    max_delay = 62 + min_delay

    if not (min_delay <= delay * fs_hz <= max_delay):
        raise ValueError(
            f"Delay must be larger than {min_delay / fs_hz * 1e9:.2f} ns and smaller than {max_delay / fs_hz * 1e9:.2f} ns."
        )

    # Fraction of sample of required shift of fine delay
    # shifted by -0.5 samples, so that it's in optimal range of -0.5, 0.5
    mu = ((delay * fs_hz) % 1.0) - 0.5

    # Calculate the FIR filter for the fine delay. Only the first 8 taps are used, the other are set to zero
    fir, tau_g = fractional_delay_filter(mu, fs_hz, f_bw_hz, 8, beta)
    fir = np.concatenate((fir, np.zeros(32)))

    # The coarse (port) delay
    # We subtract the group delay introduced by the FIR filter
    port_delay = delay - tau_g / fs_hz

    return port_delay, fir, tau_g / fs_hz


def get_signal_calibration(
    delay: float, fs_hz: float, f_bw_hz: float = 900e6, beta: float = 6.0
) -> SignalCalibration:
    """Return a SignalCalibration object to get the given delay
    on a HDAWG output.

    Args:
        delay (float): The desired delay in seconds
        fs_hz (float): sampling rate in Hz
        f_bw_hz (float, optional): cutoff frequency in Hz (must be 0 < f_bw_hz < fs_hz/2)
                                   Defaults to 900e6.
        beta (float, optional): Kaiser window beta (≈5–8 typical). Defaults to 6.0.

    Returns:
        SignalCalibration: _description_
    """

    port_delay, fir, _ = get_delay_settings(delay, fs_hz, f_bw_hz, beta)

    return SignalCalibration(
        precompensation=Precompensation(FIR=FIRCompensation(fir)), port_delay=port_delay
    )
