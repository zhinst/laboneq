# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler

plt.rcParams.update(
    {
        "font.weight": "light",
        "axes.labelweight": "light",
        "axes.titleweight": "normal",
        "axes.prop_cycle": cycler(color=["#006699", "#FF0000", "#66CC33", "#CC3399"]),
        "svg.fonttype": "none",  # Make text editable in SVG
        "text.usetex": False,
    }
)


def plot_pulse(pulse):
    """
    Plots I, Q, amplitude, and phase of a pulse waveform.

    Args:
        pulse (object): The input pulse object to be plotted.

    Returns:
        None. This function only generates a plot.
    """

    fig, axs = plt.subplots(4)
    fig.suptitle("Waveform")

    ylab1 = [r"$\mathcal{I}$", r"$\mathcal{Q}$", r"$\mathcal{A}$", r"$\Phi$"]

    for i in range(4):
        axs[i].set_ylabel(ylab1[i])

    axs[0].plot(pulse.t * 1e9, pulse.i_wave)

    axs[1].plot(pulse.t * 1e9, pulse.q_wave)

    axs[2].plot(pulse.t * 1e9, pulse.amplitudes)

    axs[3].plot(pulse.t * 1e9, pulse.phases)

    for ax in axs.flat:
        ax.set(xlabel="$t$ / ns")

    for ax in axs:
        ax.label_outer()

    plt.show()


def plot_pulse_iq(pulse):
    """
    Plots I and Q components of a pulse waveform.

    Args:
        pulse (object): The input pulse object to be plotted.

    Returns:
        None. This function only generates a plot.
    """

    fig, axs = plt.subplots(2)
    fig.suptitle(r"$\mathcal{I}$ and $\mathcal{Q}$")

    ylab1 = [r"$\mathcal{I}$", r"$\mathcal{Q}$"]

    for i in range(2):
        axs[i].set_ylabel(ylab1[i])

    axs[0].plot(pulse.t * 1e9, pulse.i_wave)

    axs[1].plot(pulse.t * 1e9, pulse.q_wave)

    for ax in axs.flat:
        ax.set(xlabel="$t$ / ns")

    for ax in axs:
        ax.label_outer()

    plt.show()


def plot_pulse_amp_phi(pulse):
    """
    Plots the amplitude and phase of a pulse waveform.

    Args:
        pulse (object): The input pulse object to be plotted.

    Returns:
        None. This function only generates a plot.
    """

    fig, axs = plt.subplots(2)
    fig.suptitle("Amplitude and Phase")

    ylab1 = [r"$\mathcal{A}$", r"$\Phi$"]

    for i in range(2):
        axs[i].set_ylabel(ylab1[i])

    axs[0].plot(pulse.t * 1e9, pulse.amplitudes)

    axs[1].plot(pulse.t * 1e9, pulse.phases)

    for ax in axs.flat:
        ax.set(xlabel="$t$ / ns")

    for ax in axs:
        ax.label_outer()

    plt.show()


def plot_excitation_profile(state_mat, offs):
    """
    Plots the excitation profile of a pulse on a range of qubits of different frequencies

    Args:
        state_mat (np.ndarray): A 3 x len(offs) matrix representing the states of qubits.
        offs (np.ndarray): A range of offsets.

    Returns:
        None. This function only generates a plot.
    """

    ylab2 = ["$x$", "$y$", "$z$", r"$\sqrt{x^2+y^2}$"]

    fig, axs = plt.subplots(4)
    for i in range(4):
        axs[i].set_ylabel(ylab2[i])
    fig.suptitle("Excitation profile")
    for i in range(3):
        axs[i].plot(offs / (2 * np.pi) / 1e6, state_mat[i, :])

    axs[3].plot(
        offs / (2 * np.pi) / 1e6, np.sqrt(state_mat[0, :] ** 2 + state_mat[1, :] ** 2)
    )

    for ax in axs.flat:
        ax.set(xlabel="Qubit offset / (MHz)")

    for ax in axs:
        ax.label_outer()

    plt.show()
