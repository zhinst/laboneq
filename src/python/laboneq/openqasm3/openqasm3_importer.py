# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Callable

import re
import openpulse
from openpulse import ast
from laboneq.core.path import remove_logical_signal_prefix
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
)
from laboneq.dsl.experiment import Experiment, Section
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.quantum.quantum_operations import QuantumOperations
from laboneq.openqasm3 import options as exp_options
from laboneq.openqasm3.gate_store import _GateStoreQuantumOperations
from laboneq.openqasm3.namespace import (
    NamespaceStack,
)
from laboneq.openqasm3 import device
from laboneq.openqasm3.openqasm_error import OpenQasmException
from laboneq.openqasm3.results import ExternResult
from laboneq.openqasm3.visitor import TranspilerVisitor

if TYPE_CHECKING:
    from laboneq.openqasm3.gate_store import GateStore


def _unwrap_qubit_register(name: str, size: int) -> list[str]:
    """Unwrap a QASM qubit register into a list of single qubits."""
    # Qubit register has a special convention
    # Hopefully a better solution is added later
    return [f"{name}[{idx}]" for idx in range(size)]


def define_qubit_names(qubits: dict) -> dict[str, str]:
    # NOTE: Unwrap qubit registers in qubit mapping
    #       Importer relies on 1-1 mapping between qasm qubit and DSL qubit
    # TODO: Currently qubit register argument not supported in Importer, but this
    #       unwraps them so the registers must exist in the mapping once the feature
    #       is supported
    # TODO: Multiple handles cannot point to same qubit e.g. $q and q0 same qubit
    # TODO: Replace with proper qubit register
    out = {}
    for k, q in (qubits or {}).items():
        if isinstance(q, Sequence):  # Qubit register (QuantumOperations)
            for name, qubit in zip(_unwrap_qubit_register(name=k, size=len(q)), q):
                if name in out:
                    raise OpenQasmException(f"Duplicate qubit definition {name}")
                out[name] = qubit  # noqa: PERF403
        else:
            if k in out:
                raise OpenQasmException(f"Duplicate qubit definition {k}")
            out[k] = q
    return out


class OpenQasm3Importer:
    def __init__(
        self,
        qops: QuantumOperations | None = None,
        qubits: dict[str, QuantumElement] | None = None,
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable | str] | None = None,
    ):
        # NOTE: still need to unpack register using naming convention "qreq[0]"
        self.qubits = define_qubit_names(qubits or {})
        self.qops = qops
        self.supplied_inputs = inputs
        self.supplied_externs = externs
        self.scope = NamespaceStack()
        for k in self.qubits:
            if k.startswith("$"):
                self.scope.current.declare_qubit(k)

    @classmethod
    def from_gate_store(
        cls: OpenQasm3Importer,
        gate_store: GateStore,
        qubits: dict[str, QuantumElement] | None = None,
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable] | None = None,
    ) -> OpenQasm3Importer:
        externs = externs or {}
        ports = gate_store.ports or {}
        waveforms = gate_store.waveforms or {}
        inputs = inputs or {}
        inp = inputs | waveforms
        return cls(
            qops=_GateStoreQuantumOperations(gate_store, qubits),
            qubits=qubits,
            inputs=inp,
            externs=externs | ports | waveforms,
        )

    def __call__(
        self,
        text: str,
    ) -> Section:
        tree = self.program_to_ast(text)
        try:
            return self.transpile(tree, uid_hint="root")
        except OpenQasmException as e:
            e.source = text
            raise

    def program_to_ast(self, text: str) -> ast.Program:
        """Convert OpenQASM program into an AST tree."""
        text = self._preprocess(text)
        tree = openpulse.parse(text)
        assert isinstance(tree, ast.Program)
        return tree

    def _workaround_extern_port(self, text: str) -> str:
        # NOTE: 'extern port' declaration is not yet supported by the openpulse parser.
        pattern = r"extern port (\S+);"
        return re.sub(pattern, r"port \1;", text)

    def _preprocess(self, text: str) -> str:
        return self._workaround_extern_port(text)

    def transpile(
        self,
        parent: ast.Program | ast.Box | ast.ForInLoop,
        uid_hint="",
    ) -> Section:
        visitor = TranspilerVisitor(
            qops=self.qops,
            qubits=self.qubits,
            inputs=self.supplied_inputs,
            externs=self.supplied_externs,
            namespace=self.scope,
        )
        return visitor.transpile(parent, uid_hint=uid_hint)


def _port_from_logical_signal_path(ls: str) -> device.Port:
    """Turn absolute logical signal path into a port."""
    ls = remove_logical_signal_prefix(ls)
    parts = ls.split("/")
    qubit = parts[0]
    return device.port(qubit, ls)


def exp_from_qasm(
    program: str,
    qubits: dict[str, QuantumElement],
    gate_store: GateStore,
    inputs: dict[str, Any] | None = None,
    externs: dict[str, Callable[..., ExternResult | Any]] | None = None,
    count: int = 1,
    averaging_mode: AveragingMode = AveragingMode.CYCLIC,
    acquisition_type: AcquisitionType | None = None,
    reset_oscillator_phase: bool = False,
) -> Experiment:
    """
    !!! version-changed "Deprecated in version 2.43.0"
        Use [OpenQASMTranspiler]() instead.

    Create an experiment from an OpenQASM program.

    Arguments:
        program:
            OpenQASM program
        qubits:
            Map from OpenQASM qubit names to LabOne Q DSL QuantumElement objects
        gate_store:
            Map from OpenQASM gate names to LabOne Q DSL Gate objects
        inputs:
            Inputs to the OpenQASM program.
        externs:
            Extern functions for the OpenQASM program.
        count:
            The number of acquire iterations.
        averaging_mode:
            The mode of how to average the acquired data.
        acquisition_type:
            The type of acquisition to perform.

            The acquisition type may also be specified within the
            OpenQASM program using `pragma zi.acquisition_type raw`,
            for example.

            If an acquisition type is passed here, it overrides
            any value set by a pragma.

            If the acquisition type is not specified, it defaults
            to [AcquisitionType.INTEGRATION]().
        reset_oscillator_phase:
            When true, reset all oscillators at the start of every
            acquisition loop iteration.

    Returns:
        The experiment generated from the OpenQASM program.
    """
    warnings.warn(
        "`exp_from_qasm()` is deprecated. Use `OpenQASMTranspiler.experiment()` instead.",
        FutureWarning,
        stacklevel=2,
    )
    from laboneq.dsl.quantum import QPU
    from laboneq.openqasm3.transpiler import OpenQASMTranspiler

    transpiler = OpenQASMTranspiler(
        qpu=QPU(
            qubits=list(qubits.values()),
            quantum_operations=_GateStoreQuantumOperations(gate_store, qubits),
        )
    )
    ports = gate_store.ports or {}
    ports = {k: _port_from_logical_signal_path(v) for k, v in ports.items()}
    externs = externs or {}
    waveforms = gate_store.waveforms or {}
    # GateStore waveforms can only be inputs (array / Pulse)
    inputs = inputs or {}
    inp = inputs | waveforms
    return transpiler.experiment(
        program=program,
        qubit_map=qubits,
        inputs=inp,
        externs=externs | ports,
        options=exp_options.SingleProgramOptions(
            count=count,
            averaging_mode=averaging_mode,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=reset_oscillator_phase,
        ),
    )


def exp_from_qasm_list(
    programs: list[str],
    qubits: dict[str, QuantumElement],
    gate_store: GateStore,
    inputs: dict[str, Any] | None = None,
    externs: dict[str, Callable[..., ExternResult | Any]] | None = None,
    count: int = 1,
    averaging_mode: AveragingMode = AveragingMode.CYCLIC,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
    reset_oscillator_phase: bool = False,
    repetition_time: float = 1e-3,
    batch_execution_mode: str = "pipeline",
    do_reset: bool = False,
    add_measurement: bool = True,
    pipeline_chunk_count: int | None = None,
) -> Experiment:
    """
    !!! version-changed "Deprecated in version 2.43.0"
        Use [OpenQASMTranspiler]() instead.

    Process a list of openQASM programs into a single LabOne Q experiment that
    executes the QASM snippets sequentially.

    At this time, the QASM programs should not include any measurements. By default, we automatically
    append a measurement of all qubits to the end of each program.
    This behavior may be changed by specifying `add_measurement=False`.

    The measurement results for each qubit are stored in a handle named
    `f'meas{qasm_qubit_name}'` where `qasm_qubit_name` is the key specified for the
    qubit in the `qubits` parameter.

    Optionally, a reset operation on all qubits is prepended to each program. The
    duration between the reset and the final readout is fixed and must be specified as
    `repetition_time`. It must be chosen large enough to accommodate the longest of the
    programs. The `repetition_time` parameter is also required if the resets are
    disabled. In a future version we hope to make an explicit `repetition_time` optional.

    For the measurement we require the gate store to be loaded with a `measurement`
    gate. Similarly, the optional reset requires a `reset` gate to be available.

    Note that using `set_frequency` or specifying the acquisition type via a
    `pragma zi.acquisition_type` statement within an OpenQASM program is not
    supported by `exp_from_qasm_list`. It will log a warning if these are encountered.

    Arguments:
        programs:
            the list of the QASM snippets
        qubits:
            Map from OpenQASM qubit names to LabOne Q DSL QuantumElement objects
        gate_store:
            Map from OpenQASM gate names to LabOne Q DSL Gate objects
        inputs:
            Inputs to the OpenQASM program.
        externs:
            Extern functions for the OpenQASM program.
        count:
            The number of acquire iterations.
        averaging_mode:
            The mode of how to average the acquired data.
        acquisition_type:
            The type of acquisition to perform.
        reset_oscillator_phase:
            When true, reset all oscillators at the start of every
            acquisition loop iteration.
        repetition_time:
            The length that any single program is padded to.
        batch_execution_mode:
            The execution mode for the sequence of programs. Can be any of the following:

            - "nt": The individual programs are dispatched by software.
            - "pipeline": The individual programs are dispatched by the sequence pipeliner.
            - "rt": All the programs are combined into a single real-time program.

            "rt" offers the fastest execution, but is limited by device memory.
            In comparison, "pipeline" introduces non-deterministic delays between
            programs of up to a few 100 microseconds. "nt" is the slowest.
        do_reset:
            If `True`,  an active reset operation is added to the beginning of each program.
        add_measurement:
            If `True`, add measurement at the end for all qubits used.
        pipeline_chunk_count:
            The number of pipeline chunks to divide the experiment into.

            The default chunk count is equal to the number of programs, so that there is one
            program per pipeliner chunk. Future versions of LabOne Q may use a more
            sophisticated default based on the program sizes.

            Currently the number of programs must be a multiple of the chunk count so that
            there are the same number of programs in each chunk. This limitation will be
            removed in a future release of LabOne Q.

            A `ValueError` is raised if the number of programs is not a multiple of the
            chunk count.

    Returns:
        The experiment generated from the OpenQASM programs.
    """
    warnings.warn(
        "`exp_from_qasm_list()` is deprecated. Use `OpenQASMTranspiler.batch_experiment()` instead.",
        FutureWarning,
        stacklevel=2,
    )
    from laboneq.dsl.quantum import QPU
    from laboneq.openqasm3.transpiler import OpenQASMTranspiler

    transpiler = OpenQASMTranspiler(
        qpu=QPU(
            qubits=list(qubits.values()),
            quantum_operations=_GateStoreQuantumOperations(gate_store, qubits),
        )
    )
    ports = gate_store.ports or {}
    ports = {k: _port_from_logical_signal_path(v) for k, v in ports.items()}
    externs = externs or {}
    waveforms = gate_store.waveforms or {}
    # GateStore waveforms can only be inputs (array / Pulse)
    inputs = inputs or {}
    inp = inputs | waveforms
    return transpiler.batch_experiment(
        programs=programs,
        qubit_map=qubits,
        inputs=inp,
        externs=externs | ports,
        options=exp_options.MultiProgramOptions(
            count=count,
            averaging_mode=averaging_mode,
            acquisition_type=acquisition_type,
            reset_oscillator_phase=reset_oscillator_phase,
            repetition_time=repetition_time,
            batch_execution_mode=batch_execution_mode,
            add_reset=do_reset,
            add_measurement=add_measurement,
            pipeline_chunk_count=pipeline_chunk_count,
        ),
    )
