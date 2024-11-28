# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import numpy as np
import scipy as sp

from laboneq.contrib.bloch_simulator_pulse_plotter.plotter.plot_funs import (
    plot_excitation_profile,
    plot_pulse,
    plot_pulse_amp_phi,
    plot_pulse_iq,
)
from laboneq.contrib.bloch_simulator_pulse_plotter.pulse_simulator.bloch_simulator import (
    pulse_sim,
)
from laboneq.core.utilities.pulse_sampler import sample_pulse


def pulse_update(
    my_pulse,
    flip_angle: float | None = None,
    spectral_window: float | None = None,
    pulse_parameters: dict | None = None,
):
    """
    Updates a pulse object with new properties and returns the updated object.

    Args:
        my_pulse (object): The input pulse object to be updated.
        flip_angle (float | None, optional): The desired flip angle for the pulse in degrees. Defaults to None (if not specified, the flip angle will be set to 180 degrees, corresponding to a pi-pulse)
        spectral_window (float | None, optional): The desired spectral window for the pulse in Hz. Defaults to None.
        pulse_parameters (dict[str, float | int | str] | None, optional): A dictionary containing any additional parameters for the pulse function. Defaults to None.

    Returns:
        object: The updated pulse object.
    """

    # get numeric waveform
    numeric_waveform = sample_pulse(
        pulse_function=my_pulse.function,
        signal_type="iq",
        length=my_pulse.length,
        sampling_rate=2e9,
        amplitude=my_pulse.amplitude,
        pulse_parameters=pulse_parameters,
    )

    # get i and q quadratures
    my_pulse.i_wave = numeric_waveform.get("samples_i")
    my_pulse.q_wave = numeric_waveform.get("samples_q")

    # calculate amplitude and phase
    my_pulse.amplitudes = np.sqrt(my_pulse.i_wave**2 + my_pulse.q_wave**2)
    my_pulse.phases = np.arctan2(my_pulse.q_wave, my_pulse.i_wave)

    # get flip angle and peak amplitude (flip angle should be user defined)
    if flip_angle is not None:
        my_pulse.flip_angle = flip_angle
    else:
        my_pulse.flip_angle = 180

    if spectral_window is not None:
        my_pulse.spectral_window = spectral_window
    else:
        my_pulse.spectral_window = 200e6

    area = sp.integrate.simpson(my_pulse.amplitudes)
    my_pulse.peak_amplitude = (
        2
        * np.pi
        * (len(my_pulse.i_wave) / area)
        * (my_pulse.flip_angle / 360)
        / my_pulse.length
    )

    # set time axis
    my_pulse.t = np.linspace(0, my_pulse.length, len(my_pulse.i_wave))

    return my_pulse


def pulse_inspector(
    pulse,
    iq: bool | None = None,
    amp_phi: bool | None = None,
    response: bool | None = None,
    initial_state: np.ndarray | None = None,
):
    """
    Plot the excitation profile of a virtual qubit after applying the pulse as a function of qubit frequency.

    Args:
        pulse (object): The input pulse waveform to be analyzed. Must be an object.
        iq (bool | None, optional): If True, plot the I-Q diagram of the pulse. Defaults to None. If no flag is specified, this flag is set to True.
        amp_phi (bool | None, optional): If True, plot the amplitude-phase diagram of the pulse. Defaults to None.
        response (bool | None, optional): If True, plot the excitation profile of a qubit after applying the pulse. Defaults to None.
        initial_state (np.ndarray | None, optional): The initial state of the virtual qubit as a Bloch vector. Must be a 1D numpy array of length 3. Defaults to None.

    Returns:
        None. This function only generates plots.

    Raises:
        ValueError: If the input pulse is None.
    """

    if pulse is None:
        raise ValueError("waveform must be provided")

    if iq is None and amp_phi is None and response is None:
        iq = True
        print("Nothing has been specified; iq flag was set to True.")

    if iq and amp_phi:
        plot_pulse(pulse)

    if iq:
        plot_pulse_iq(pulse)

    if amp_phi:
        plot_pulse_amp_phi(pulse)

    if response:
        if initial_state is None:
            initial_state = np.array([0.0, 0.0, 1.0])
            print(
                f"Initial state has not been provided, Initial state is set to |0> := {initial_state}"
            )
        n_qubits = 1000
        out_state, offs = pulse_sim(pulse, n_qubits, initial_state)
        plot_excitation_profile(out_state, offs)
