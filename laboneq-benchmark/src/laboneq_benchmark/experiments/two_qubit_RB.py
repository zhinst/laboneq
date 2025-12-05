# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

# Qiskit
from qiskit import qasm3, transpile
from qiskit_experiments.library import randomized_benchmarking

# LabOne Q
from laboneq import openqasm3
from laboneq.dsl.device import DeviceSetup
from laboneq.dsl.quantum import QPU


def two_qubit_RB(
    device_setup: DeviceSetup,
    qpu: QPU,
    number_average: int = 2**12,
    sequence_exponent: int = 5,
    chunk_count: int = 4,
    n_sequences_per_length: int = 40,
):
    # wrap the experiment definition in an inner function to be able to use it as a callable
    def inner():
        length = 2**sequence_exponent
        # Use Qiskit Experiment Library to Generate RB
        rb2_qiskit_circuits = randomized_benchmarking.StandardRB(
            physical_qubits=[0, 1],
            lengths=[length],
            num_samples=n_sequences_per_length,
        ).circuits()
        for circuit in rb2_qiskit_circuits:
            circuit.remove_final_measurements()
        # Choose basis gates
        rb2_transpiled_circuits = transpile(
            rb2_qiskit_circuits, basis_gates=["id", "sx", "x", "rz", "cx"]
        )

        rb2_program_list = [qasm3.dumps(circuit) for circuit in rb2_transpiled_circuits]

        # Instantiate OpenQASMTranspiler from the QPU
        transpiler = openqasm3.OpenQASMTranspiler(qpu)

        # Define transpiler options
        options = openqasm3.MultiProgramOptions()
        options.count = number_average
        options.batch_execution_mode = "pipeline"
        options.pipeline_chunk_count = chunk_count
        options.add_measurement = (
            True  # adds the measurement operations which were removed above
        )
        # Set the format of the result handles to align with the handles
        # defined in our quantum operations:
        options.add_measurement_handle = "{qubit.uid}/result"

        # qubits = device_setup.qubits[:1]
        # Create the Experiment
        rb2_exp = transpiler.batch_experiment(
            programs=rb2_program_list,
            qubit_map={"q": [qubit for qubit in device_setup.qubits.values()]},
            options=options,
        )

        return rb2_exp

    return inner
