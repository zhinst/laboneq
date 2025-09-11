# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

# additional imports needed for Clifford gate calculation
from laboneq.contrib.example_helpers.randomized_benchmarking_helper import (
    clifford_parametrized,
    generate_play_rb_pulses,
    make_pauli_gate_map,
)
from laboneq.dsl import LinearSweepParameter, SweepParameter
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    SectionAlignment,
)
from laboneq.dsl.experiment import (
    Experiment,
    ExperimentSignal,
    pulse_library,
)
from laboneq.dsl.quantum import QuantumElement

from .pulses import (
    integration_kernel,
    readout_pulse,
)

# Adjust Pulse Parameters for Clifford Gates

# Define and prepare the basic gate set and the pulse objects corresponding to them
pulse_reference = pulse_library.drag
pulse_parameters = {"sigma": 1 / 3, "beta": 0.4}
pulse_length = 32e-9

gate_map = make_pauli_gate_map(
    pi_pulse_amp=0.4,
    pi_half_pulse_amp=0.21,
    excitation_length=pulse_length,
    pulse_factory=pulse_reference,
    pulse_kwargs=pulse_parameters,
)


# define a convenience function to generate the RB sequences
def sweep_rb_pulses(
    sequence_length: int | SweepParameter | LinearSweepParameter,
    exp: Experiment,
    signal: str,
    cliffords,
    gate_map,
    rng,
):
    generate_play_rb_pulses(
        exp=exp,
        signal=signal,
        seq_length=sequence_length,
        cliffords=cliffords,
        gate_map=gate_map,
        rng=rng,
    )


# define the RB experiment
def rb_parallel(
    qubits: list[QuantumElement],
    number_average: int = 2**12,
    sequence_exponent: int = 8,
    # max_sequence_exponent: int = 8,
    chunk_count: int = 1,
    n_sequences_per_length: int = 20,
    do_readout: bool = False,
    prng_seed: int = 42,
):
    # wrap the experiment definition in an inner function to be able to use it as a callable
    def inner():
        # construct the sweep over sequence length as powers of 2 of the sequence exponent
        sequence_length = 2**sequence_exponent
        sequence_sweep = SweepParameter(
            values=np.array(range(1, n_sequences_per_length + 1))
        )

        # we are using fixed timing, where the maximum duration is determined by the maximum sequence length
        max_seq_duration = (2**sequence_exponent + 1) * 3 * pulse_length

        prng = np.random.default_rng(seed=prng_seed)

        exp_rb = Experiment(
            uid="RandomizedBenchmark",
            signals=[
                signal
                for signal_list in [
                    [
                        ExperimentSignal(
                            f"drive_{qubit.uid}", map_to=qubit.signals["drive"]
                        ),
                        ExperimentSignal(
                            f"measure_{qubit.uid}", map_to=qubit.signals["measure"]
                        ),
                        ExperimentSignal(
                            f"acquire_{qubit.uid}", map_to=qubit.signals["acquire"]
                        ),
                    ]
                    for qubit in qubits
                ]
                for signal in signal_list
            ],
        )

        # outer loop - real-time, cyclic averaging in discrimination mode
        with exp_rb.acquire_loop_rt(
            uid="rb_shots",
            count=number_average,
            averaging_mode=AveragingMode.CYCLIC,
            acquisition_type=AcquisitionType.DISCRIMINATION,
        ):
            # inner loop - sweep over sequence lengths
            with exp_rb.sweep(parameter=sequence_sweep, chunk_count=chunk_count):
                for qubit in qubits:
                    with exp_rb.section(
                        uid=f"{qubit.uid}_drive_{sequence_sweep}",
                        length=max_seq_duration,
                        alignment=SectionAlignment.RIGHT,
                    ):
                        sweep_rb_pulses(
                            sequence_length,
                            exp_rb,
                            f"drive_{qubit.uid}",
                            clifford_parametrized,
                            gate_map,
                            prng,
                        )
                    # readout and data acquisition
                    if do_readout:
                        with exp_rb.section(
                            uid=f"{qubit.uid}_measure_{sequence_sweep}",
                            play_after=f"{qubit.uid}_drive_{sequence_sweep}",
                        ):
                            exp_rb.measure(
                                measure_pulse=readout_pulse,
                                measure_signal=f"measure_{qubit.uid}",
                                acquire_signal=f"acquire_{qubit.uid}",
                                handle=f"{qubit.uid}_rb_results",
                                integration_kernel=integration_kernel,
                                reset_delay=qubit.parameters.user_defined[
                                    "reset_delay_length"
                                ],
                            )
                            exp_rb.reserve(signal=f"drive_{qubit.uid}")

        return exp_rb

    return inner
