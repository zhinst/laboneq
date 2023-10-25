# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Helper Functions for Randomized Benchmarking examples, containing
    - Definitions of the basic Clifford gates
    - Functionality to calculate the recovery gate
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy import random as nprnd
from numpy import typing as npt
from scipy.linalg import expm as matrix_exponential
from typing_extensions import (
    ParamSpec,  # FIXME: Reconsider when python>=3.10 is enforced.
)

from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.pulse_library import PulseFunctional, gaussian
from laboneq.simple import Experiment

P = ParamSpec("P")


class PauliGateMap(dict[str, PulseFunctional]):
    """Mapping from gate label to a sequence of `PulseFunctional`s.

    Use `.make` to construct this object.
    """

    @classmethod
    def make(
        cls,
        excitation_length: float,
        pi_pulse_amp: float,
        pi_half_pulse_amp: float,
        pulse_factory: Callable[P, PulseFunctional] = gaussian,
        pulse_kwargs: P.kwargs | None = None,
    ):
        """Construct a mapping from Pauli gate names to gate representations as
        `PulseFunctional`s.

        The gate set consists of identity, X and Y pi gates, as well as +-X/2 and +-Y/2 gates.
        Gaussian pulses are sampled for each of these fundamental gates.

        The pulses obtained via `pulse_factory` will be rotated in complex plane to
        implement the gates.

        Args:
            excitation_length: Duration of a gate.
            pi_pulse_amp: Amplitude of the pi gates.
            pi_half_pulse_amp: Amplitude of the pi/2 gates.
            pulse_factory: A callable to generate `PulseFunctional`s.  Typically,
                functions marked with `pulse_library.register_pulse_functional` should be
                provided here.
            pulse_kwargs: A dictionary of arguments to pass into `pulse_factory` as
                keyword arguments.  Must not contain "amplitude" or "length" in keys.

        Returns:
            gate_map: Dictionary of gates mapping gate labels to their pulse representations.
        """
        pulse_kwargs = {} if pulse_kwargs is None else pulse_kwargs

        forbidden_kws = "amplitude", "length"
        found_forbidden_kws = [k for k in forbidden_kws if k in pulse_kwargs]
        if found_forbidden_kws != []:
            raise ValueError(
                f"Any of {forbidden_kws} can not be used in the keyword arguments to "
                f"`pulse_factory` (found {found_forbidden_kws})."
            )

        pi = np.pi
        gate_amp_phase_alist = (
            ("I", 0, 0),
            ("X", pi_pulse_amp, 0),
            ("Y", pi_pulse_amp, pi / 2),
            ("X/2", pi_half_pulse_amp, 0),
            ("Y/2", pi_half_pulse_amp, pi / 2),
            ("-X/2", -pi_half_pulse_amp, 0),
            ("-Y/2", -pi_half_pulse_amp, pi / 2),
        )
        mapping = {
            gate: pulse_factory(
                amplitude=amp * np.exp(1j * phase),
                length=excitation_length,
                **pulse_kwargs,
            )
            for gate, amp, phase in gate_amp_phase_alist
        }
        return cls(mapping)


make_pauli_gate_map = PauliGateMap.make


# Definition of all gates in the Clifford group in terms of Pauli gates
clifford_parametrized = (
    ("I",),
    ("Y/2", "X/2"),
    ("-X/2", "-Y/2"),
    ("X",),
    ("-Y/2", "-X/2"),
    ("X/2", "-Y/2"),
    ("Y",),
    ("-Y/2", "X/2"),
    ("X/2", "Y/2"),
    ("X", "Y"),
    ("Y/2", "-X/2"),
    ("-X/2", "Y/2"),
    ("Y/2", "X"),
    ("-X/2",),
    ("X/2", "-Y/2", "-X/2"),
    ("-Y/2",),
    ("X/2",),
    ("X/2", "Y/2", "X/2"),
    ("-Y/2", "X"),
    ("X/2", "Y"),
    ("X/2", "-Y/2", "X/2"),
    ("Y/2",),
    ("-X/2", "Y"),
    ("X/2", "Y/2", "-X/2"),
)


_pauli_matrices = {
    "x": np.array([[0, 1], [1, 0]]),
    "y": np.array([[0, -1j], [1j, 0]]),
    "z": np.array([[1, 0], [0, -1]]),
}


def pauli(axis: str):
    """Returns the Pauli matrix for a given axis

    Args:
        axis: One of "x", "y", "z"

    Returns:
        Respective Pauli matrix
    """
    return _pauli_matrices[axis]


def rot_matrix(angle=np.pi, axis="x"):
    """General definition of unitary rotation matrix for a single two-level qubit

    Args:
        angle: angle of rotation
        axis: string, either "x", "y", or "z"

    Returns:
        Rotation matrix
    """
    return matrix_exponential(-1j * angle / 2 * pauli(axis))


def mult_gates(gate_list):
    """Multiply a variable number of gates / matrices

    Args:
        gate_list: List of gates that shall be multiplied

    Returns:
        Result of the matrix multiplication
    """
    if len(gate_list) == 1:
        return gate_list[0]
    else:
        return np.linalg.multi_dot(gate_list)


# Matrix representation of all elementary gates used to generate the Clifford gates
elem_gates = {
    "I": np.array([[1, 0], [0, 1]]),
    "X": rot_matrix(np.pi, "x"),
    "Y": rot_matrix(np.pi, "y"),
    "X/2": rot_matrix(np.pi / 2, "x"),
    "Y/2": rot_matrix(np.pi / 2, "y"),
    "-X/2": rot_matrix(-np.pi / 2, "x"),
    "-Y/2": rot_matrix(-np.pi / 2, "y"),
}

# Set of Clifford gates, specified as sequence of gates in matrix form
clifford_matrices = [
    tuple(elem_gates[gate] for gate in gates) for gates in clifford_parametrized
]

# Set of Clifford gates, specified as single matrix per Clifford gate
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


def calculate_inverse_clifford_index(
    index_list, clifford_gates: list[np.ndarray] = clifford_gates
):
    """Given a set of gates indexed with integers in `index_list`, calculate the
    inverse of the cascaded gates and return the index corresponding to the result.  The
    calculated matrix is searched up to a global phase in `clifford_gates` and the index of
    the matching entry is returned.

    Args:
        index_list: a list containing the indices of the clifford sequence to be inverted
        clifford_list: a list containing the set of clifford gates

    Returns:
        recovery: index of the recovery gate
    """
    # matrix representation of the full Clifford sequence
    seq_gate = mult_gates([clifford_gates[ind] for ind in index_list])
    # recovery gate - inverse of full sequence
    rec_gate = np.linalg.inv(seq_gate)
    # index of recovery gate (up to global phase)
    recovery = int(match_up_to_phase(rec_gate, clifford_gates))

    return recovery


def calculate_clifford_sequence_indices(
    seq_length: int,
    cliffords: tuple[tuple[str]],
    elementary_gates: dict[str, np.ndarray] = elem_gates,
    rng: nprnd.Generator | None = None,
) -> npt.NDArray[int]:
    """Generate a set of randomly selected indices and an index for a recovery Clifford
    gate to use with RB sequence.

    Args:
        seq_length: Length of the RB sequence, excluding recovery gate.
        cliffords: Sequence of Clifford gates represented by a list of elementary gate
            names (∈ {'I', 'X', 'Y', '±X/2', '±Y/2'}).
        elementary_gates: Mapping from elementary gate names to their matrix representations.
        rng: A `numpy.random.Generator` object to use when selecting the clifford gates.
            If `None` a new one will be created via `numpy.random.default_rng(42)`.
    """
    rng = nprnd.default_rng(42) if rng is None else rng

    # +1 to allocate space for the recovery gate.
    clifford_indices = np.empty(shape=seq_length + 1, dtype=int)
    clifford_indices[:-1] = rng.integers(0, 23, size=seq_length)

    if cliffords is clifford_parametrized:
        # shortcut for precomputed defaults
        computed_gates = clifford_gates
    else:
        computed_gates = [
            mult_gates([elementary_gates[gate] for gate in gates])
            for gates in cliffords  # "gate" ≡ pauli gate
        ]

    # Last gate is the recovery gate.
    clifford_indices[-1] = calculate_inverse_clifford_index(
        clifford_indices[:-1], clifford_gates=computed_gates
    )
    return clifford_indices


def generate_play_rb_pulses(
    exp: Experiment,
    signal: str,
    seq_length: int,
    cliffords: tuple[tuple[str]],
    gate_map: dict[str, Pulse],
    elementary_gates: dict[str, np.ndarray] = elem_gates,
    rng: nprnd.Generator | None = None,
) -> None:
    """Generate a RB sequence using the experiment handle `exp`.  Mutates `exp`.

    A RB sequence consist of randomly chosen `seq_length` number of Clifford gates
    and a final recovery gate.

    Args:
        exp: LabOne Q experiment.
        signal: Experiment signal line where pulses are played.
        seq_length: Length of the RB sequence, excluding recovery gate.
        cliffords: Sequence of basic Clifford gates.  Each Clifford gate is represented as
            a tuple of strings referring to a basic gate in `gate_map`.
        gate_map: Dictionary of gates represented as pulses to construct the Clifford gates from.
        elementary_gates: Matrix representations of the elementary gates.
        rng: A `numpy.random.Generator` object to use when selecting the clifford gates.
            If `None` a new one will be created via `numpy.random.default_rng(42)`.
    """
    assert elementary_gates.keys() == gate_map.keys()

    clifford_indices = calculate_clifford_sequence_indices(
        seq_length, cliffords, elementary_gates, rng
    )

    for it in clifford_indices:
        clifford = cliffords[it]
        for basic_gate_name in clifford:
            basic_gate_as_pulse = gate_map[basic_gate_name]
            exp.play(signal, basic_gate_as_pulse)
