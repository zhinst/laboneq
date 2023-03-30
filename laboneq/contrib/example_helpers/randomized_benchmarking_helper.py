# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Helper Functions for Randomized Benchmarking examples, containing
    - Definitions of the basic Clifford gates
    - Functionality to calculate the recovery gate
"""


# additional imports for Clifford gate calculation
import numpy as np
from scipy.linalg import expm as matrix_exponential

from laboneq.dsl.experiment import pulse_library


def pulse_envelope(amplitude, pulse_length, phase, sigma, sample_rate):
    """Samples a Gaussian pulse

    Args:
        amplitude: scaling of the pulse, between 0 and 1
        pulse_length: length of the pulse in seconds
        phase: phase of the pulse in degree.
        sigma: ratio between standard deviation and pulse length
        sample_rate: sampling rate of the instrument playing the pulse

    Returns:
        A numpy array of lists of real and complex samples.
    """
    # length in samples, rounded to the waveform granularity of 16 samples
    length = round(sample_rate * pulse_length / 16) * 16
    x = np.linspace(-1, 1, length)
    # output is complex, where phase later determines the gate rotation axis
    y = amplitude * np.exp(-(x**2) / sigma**2 + 1j * np.deg2rad(phase))

    return np.transpose(np.array([y.real, y.imag]))


# basic gate pulse set as complex 2D arrays
def basic_gate_set(pi_amp, pi_2_amp, gate_time, sigma, sample_rate):
    """Creates and returns the basic gate pulse set.

    The gate set consists of identity, X and Y pi gates, as well as +-X/2 and +-Y/2 gates.
    Gaussian pulses are sampled for each of these fundamental gates.

    Args:
        pi_amp: Amplitude of the pi gates
        pi_2_amp: Amplitude of the pi/2 gates
        gate_time: length of a clifford gate
        sigma: ratio of the standard deviation to the gate length
        sample_rate: sampling rate of the instrument playing the pulse

    Returns:
        gate_set: Dictionary of gate names and sampled pulses with given parameters and sampling rates
    """

    gate_set = {
        "I": pulse_envelope(0, gate_time, 0, sigma=sigma, sample_rate=sample_rate),
        "X": pulse_envelope(pi_amp, gate_time, 0, sigma=sigma, sample_rate=sample_rate),
        "Y": pulse_envelope(
            pi_amp, gate_time, 90, sigma=sigma, sample_rate=sample_rate
        ),
        "X/2": pulse_envelope(
            pi_2_amp, gate_time, 0, sigma=sigma, sample_rate=sample_rate
        ),
        "Y/2": pulse_envelope(
            pi_2_amp, gate_time, 90, sigma=sigma, sample_rate=sample_rate
        ),
        "-X/2": pulse_envelope(
            -pi_2_amp, gate_time, 0, sigma=sigma, sample_rate=sample_rate
        ),
        "-Y/2": pulse_envelope(
            -pi_2_amp, gate_time, 90, sigma=sigma, sample_rate=sample_rate
        ),
    }
    return gate_set


def basic_pulse_set(gate_set):
    """Creates and returns the basic gate set as sampled complex pulses.

    Args:
        gate_set: the basic gate set as dictionary

    Returns:
        pulse_set: Dictionary of gate names and sampled complex pulse objects derived from the gate_set
    """
    pulse_set = {}
    for key in gate_set.keys():
        pulse_set[key] = pulse_library.sampled_pulse_complex(gate_set[key])

    return pulse_set


### Clifford gate definitions

# definition of all gates in the Clifford group in terms of Pauli gates
clifford_parametrized = [
    ["I"],
    ["Y/2", "X/2"],
    ["-X/2", "-Y/2"],
    ["X"],
    ["-Y/2", "-X/2"],
    ["X/2", "-Y/2"],
    ["Y"],
    ["-Y/2", "X/2"],
    ["X/2", "Y/2"],
    ["X", "Y"],
    ["Y/2", "-X/2"],
    ["-X/2", "Y/2"],
    ["Y/2", "X"],
    ["-X/2"],
    ["X/2", "-Y/2", "-X/2"],
    ["-Y/2"],
    ["X/2"],
    ["X/2", "Y/2", "X/2"],
    ["-Y/2", "X"],
    ["X/2", "Y"],
    ["X/2", "-Y/2", "X/2"],
    ["Y/2"],
    ["-X/2", "Y"],
    ["X/2", "Y/2", "-X/2"],
]

#### basic definitions to manipulate Clifford gates - needed for recovery gate calculation


def pauli(axis):
    """Returns the Pauli matrix for a given axis

    Args:
        axis: string, either "x", "y", or "z"

    Returns:
        Respective Pauli matrix
    """
    if axis == "x":
        res = np.array([[0, 1], [1, 0]])
    if axis == "y":
        res = np.array([[0, -1j], [1j, 0]])
    if axis == "z":
        res = np.array([[1, 0], [0, -1]])

    return res


def rot_matrix(angle=np.pi, axis="x"):
    """General definition of unitary rotation matrix for a single two-level qubit

    Args:
        angle: angle of rotation
        axis: string, either "x", "y", or "z"

    Returns:
        Rotation matrix
    """
    return matrix_exponential(-1j * angle / 2 * pauli(axis))


def mult_gates(gate_list, use_linalg=False):
    """Multiply a variable number of gates / matrices

    Recursive definition fastest for simple 2x2 matrices

    Args:
        gate_list: List of gates that shall be multiplied
        use_linalg: If true, the numpy.linalg library is used. Otherwise: Recursive

    Returns:
        Result of the matrix multiplication
    """
    if len(gate_list) > 1:
        if use_linalg:
            res = np.linalg.multi_dot(gate_list)
        else:
            res = np.matmul(gate_list[0], mult_gates(gate_list[1:], use_linalg=False))
    elif len(gate_list) == 1:
        res = gate_list[0]

    return res


# generate matrix representation of all elementary gates used to generate the Clifford gates
elem_gates = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": rot_matrix(np.pi, "x"),
    "Y": rot_matrix(np.pi, "y"),
    "X/2": rot_matrix(np.pi / 2, "x"),
    "Y/2": rot_matrix(np.pi / 2, "y"),
    "-X/2": rot_matrix(-np.pi / 2, "x"),
    "-Y/2": rot_matrix(-np.pi / 2, "y"),
}

# set of Clifford gates, specified as list of gates in matrix form
clifford_matrices = [
    [elem_gates[gate] for gate in gates] for gates in clifford_parametrized
]

# set of Clifford gates, specified as single matrix per Clifford
clifford_gates = [mult_gates(matrices) for matrices in clifford_matrices]


def glob_phase(phase, dim=2):
    """Global phase operator for dimensionality dim

    Args:
        phase: phase
        dim: dimensionality

    Returns:
        Global phase operator
    """
    return np.exp(1j * phase) * np.identity(dim)


def match_up_to_phase(target, gate_list, dim=2):
    """Finds the element of the gates list that best matches the target gate

    Matching is done up to a global phase of integer multiples of pi

    Args:
        target: The target gate that should be matched
        gate_list: A list of gates from which the target gate will be determined
        dim: dimensionality of the gate matrices
    """
    # set of global phase operators for integer multiples of pi
    glob_phases = [glob_phase(0, dim), glob_phase(np.pi, dim)]
    # elements of gate_list up to global phases
    gates_2 = [
        [mult_gates([gate1, gate2]) for gate2 in glob_phases] for gate1 in gate_list
    ]
    # index of gate_list that is closest to target up to global phase (using frobenius norm)
    match_index = np.argmin(
        [
            np.amin([np.linalg.norm(target - gate) for gate in gates])
            for gates in gates_2
        ]
    )

    return match_index


#### function to calculate the last gate in the sequence - recovery gate which leads back to initial state (up to global phase)
def calculate_inverse_clifford(seq_list, clifford_list=clifford_gates):
    """Calculates the final recovery gate in a sequence of Clifford gates

    Args:
        seq_list: a list containing the indices of the clifford sequence to be inverted
        clifford_list: a list containing the set of clifford gates

    Returns:
        recovery: index of the recovery gate
    """
    # matrix representation of the full Clifford sequence
    seq_gate = mult_gates([clifford_list[ind] for ind in seq_list])
    # recovery gate - inverse of full sequence
    rec_gate = np.linalg.inv(seq_gate)
    # index of recovery gate (up to global phase)
    recovery = int(match_up_to_phase(rec_gate, clifford_list))

    return recovery


### random sequences of Clifford gates


def generate_play_rb_pulses(exp, signal, seq_length, cliffords, pulse_set):
    """Generate and play RB sequence

    Generates random RB sequence, calculates recovery gate,
    samples pulses and plays them

    Args:
        exp: LabOne Q SW experiment
        signal: Experiment signal line where pulses are played
        seq_length: Length of the RB sequence, excluding recovery gate
        cliffords: List of basic Clifford gates
        pulse_set: basic pulse set
    """

    seq_list = np.random.randint(0, 23, size=seq_length)
    rec_gate = calculate_inverse_clifford(seq_list)
    np.append(seq_list, [rec_gate])

    for it in seq_list:
        for jt in range(len(cliffords[it])):
            exp.play(signal, pulse_set[cliffords[it][jt]])
