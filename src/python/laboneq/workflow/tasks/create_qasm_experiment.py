# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module provides a task to create an experiment from an OpenQASM program."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, Callable


from laboneq.openqasm3 import device
from laboneq.workflow import task
from laboneq.dsl.experiment import Experiment
from laboneq.openqasm3.options import SingleProgramOptions

if TYPE_CHECKING:
    from laboneq.dsl import quantum
    from laboneq.dsl.quantum import QuantumElement


@task
def create_qasm_experiment(
    qpu: quantum.QPU,
    program: str,
    qubit_map: dict[str, QuantumElement | list[QuantumElement]],
    inputs: dict[str, Any] | None = None,
    externs: dict[str, Callable | device.Port] | None = None,
    options: SingleProgramOptions | dict | None = None,
) -> Experiment:
    """A task to create a QASM experiment.

    This task is used to create a QASM experiment for execution on a quantum processor.

    Arguments:
        qpu:
            The quantum processing unit to create the experiment for.
        program:
            OpenQASM program.
        qubit_map:
            A map from OpenQASM qubit names to LabOne Q DSL Qubit objects.
            The values can either be a single qubit or a list of qubits in case of an qubit register.
        inputs:
            Inputs to the program.
        externs:
            A mapping for program extern definitions.

            Externs may be either functions (Python `callable`s) or
            ports on qubit signals.
        options:
            Optional settings for the LabOne Q Experiment.

            Default: [SingleProgramOptions]()

            Accepts also a dictionary with following items:

            **count**:
                The number of acquire iterations.

            **acquisition_mode**:
                The mode of how to average the acquired data.

            **acquisition_type**:
                The type of acquisition to perform.
                The acquisition type may also be specified within the
                OpenQASM program using `pragma zi.acqusition_type raw`,
                for example.
                If an acquisition type is passed here, it overrides
                any value set by a pragma.
                If the acquisition type is not specified, it defaults
                to [enums.AcquisitionType.INTEGRATION]().

            **reset_oscillator_phase**:
                When true, reset all oscillators at the start of every
                acquisition loop iteration.

    Returns:
        A LabOne Q Experiment.

    Raises:
        ValueError: Supplied qubit(s) does not exists in the QPU.
        OpenQasmException: The program cannot be transpiled.
    """
    # To avoid circular import
    from laboneq.openqasm3.transpiler import OpenQASMTranspiler

    transpiler = OpenQASMTranspiler(qpu)

    return transpiler.experiment(
        program=program,
        qubit_map=qubit_map,
        inputs=inputs,
        externs=externs,
        options=options,
    )


@task
def create_qasm_batch_experiment(
    qpu: quantum.QPU,
    programs: list[str],
    qubit_map: dict[str, QuantumElement | list[QuantumElement]],
    inputs: dict[str, Any] | None = None,
    externs: dict[str, Callable | device.Port] | None = None,
    options: SingleProgramOptions | dict | None = None,
) -> Experiment:
    """A task to create a QASM batch experiment.

    Arguments:
        qpu:
            The quantum processing unit to create the experiment for.
        programs:
            List of OpenQASM program.
        qubit_map:
            A map from OpenQASM qubit names to LabOne Q DSL Qubit objects.
            The values can either be a single qubit or a list of qubits in case of an qubit register.
        inputs:
            Inputs to the program.
        externs:
            A mapping for program extern definitions.

            Externs may be either functions (Python `callable`s) or
            ports on qubit signals.
        options:
            Optional settings for the LabOne Q Experiment.

            Default: [MultiProgramOptions]()

            Accepts also a dictionary with the following items:

            **count**:
                The number of acquire iterations.

            **acquisition_mode**:
                The mode of how to average the acquired data.

            **acquisition_type**:
                The type of acquisition to perform.
                The acquisition type may also be specified within the
                OpenQASM program using `pragma zi.acqusition_type raw`,
                for example.
                If an acquisition type is passed here, it overrides
                any value set by a pragma.
                If the acquisition type is not specified, it defaults
                to [enums.AcquisitionType.INTEGRATION]().

            **reset_oscillator_phase**:
                When true, reset all oscillators at the start of every
                acquisition loop iteration.

            **repetition_time**:
                The length that any single program is padded to.
                The duration between the reset and the final readout is fixed and must be specified as
                `repetition_time`. It must be chosen large enough to accommodate the longest of the
                programs. The `repetition_time` parameter is also required if the resets are
                disabled. In a future version we hope to make an explicit `repetition_time` optional.

            **batch_execution_mode**:
                The execution mode for the sequence of programs. Can be any of the following.

            * `nt`: The individual programs are dispatched by software.
            * `pipeline`: The individual programs are dispatched by the sequence pipeliner.
            * `rt`: All the programs are combined into a single real-time program.

            `rt` offers the fastest execution, but is limited by device memory.
            In comparison, `pipeline` introduces non-deterministic delays between
            programs of up to a few 100 microseconds. `nt` is the slowest.

            **do_reset**:
                If `True`, an active reset operation is added to the beginning of each program.

            Note: Requires `reset(qubit)` operation to be defined for each qubit.

            **add_measurement**:
                If `True`, add measurement at the end of each program for all qubits used.
                The measurement results for each qubit are stored in a handle named
                `'meas{qasm_qubit_name}'` where `qasm_qubit_name` is the key specified for the
                qubit in the `qubit_map` parameter.

            Note: Requires `measure(qubit, handle: str)` operation to be defined for each qubit.

            **pipeline_chunk_count**:
                The number of pipeline chunks to divide the experiment into.

    Returns:
        A LabOne Q Experiment.

    Raises:
        ValueError: Supplied qubit(s) does not exists in the QPU.
        OpenQasmException: The program cannot be transpiled.
    """
    # To avoid circular import
    from laboneq.openqasm3.transpiler import OpenQASMTranspiler

    transpiler = OpenQASMTranspiler(qpu)

    return transpiler.batch_experiment(
        programs=programs,
        qubit_map=qubit_map,
        inputs=inputs,
        externs=externs,
        options=options,
    )
