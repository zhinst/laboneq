# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import numpy as np


def pulse_sim(pulse, n_qubits, initial_state):
    """
    Simulates the effect of a pulse on a range of qubit offsets and returns the final state of the qubits and their frequency offsets.

    Args:
        pulse: Object produced by the pulse library.
        n_qubits (int): Number of qubits for simulation of the effect of the pulse on a range of qubit offsets.
        initial_state (numpy.ndarray): Initial state of the qubit, represented as a 3 by 1 vector corresponding to the Bloch vector of the qubit with x, y, and z components, respectively.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: The matrix of final state per qubit (3 by number of qubits) and the frequency offsets of qubits.
    """

    offs = (
        2
        * np.pi
        * np.linspace(
            -1.5 * pulse.spectral_window / 2,
            1.5 * pulse.spectral_window / 2,
            num=n_qubits,
        )
    )

    dt = pulse.t[1] - pulse.t[0]

    out_state = rot_out_magn(
        pulse.i_wave, pulse.q_wave, pulse.peak_amplitude, offs, dt, initial_state
    )

    return out_state, offs


def rot_out_magn(i_wave, q_wave, peak_amp, offs, dt, init):
    """
    Updates the state of qubits after applying a pulse, given the real and imaginary components of the pulse, the peak amplitude (in rad/sec), frequency offsets of qubits, time slice for updating the Hamiltonian (the same as t_res of the pulse), and initial state of the qubits.

    Args:
        i_wave (numpy.ndarray): The real component of the pulse.
        q_wave (numpy.ndarray): The imaginary component of the pulse.
        peak_amp (float): The peak amplitude of the pulse (in rad/sec).
        offs (numpy.ndarray): The frequency offsets of qubits.
        dt (float): The time slice for updating the Hamiltonian (the same as t_res of the pulse).
        init (numpy.ndarray): The initial state of the qubits, represented as a 3 by 1 vector.

    Returns:
        numpy.ndarray: The matrix of final state per qubit (3 by number of qubits).
    """

    qubit_state = np.zeros((3, len(offs)))
    for j in range(len(offs)):
        out = init
        for k in range(len(i_wave)):
            h = dt * np.array([peak_amp * i_wave[k], peak_amp * q_wave[k], offs[j]])
            gamma = dt * np.sqrt(
                (peak_amp * i_wave[k]) ** 2 + (peak_amp * q_wave[k]) ** 2 + offs[j] ** 2
            )
            r = np.array([[0.0, -h[2], h[1]], [h[2], 0.0, -h[0]], [-h[1], h[0], 0.0]])
            rot_mat = (
                np.identity(3)
                + (np.sin(gamma) / gamma) * r
                + ((1 - np.cos(gamma)) / (gamma**2)) * (r @ r)
            )
            out = rot_mat @ out
            qubit_state[:, j] = out

    return qubit_state
