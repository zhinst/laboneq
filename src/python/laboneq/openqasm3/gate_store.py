# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import Callable, Dict, Tuple
from itertools import takewhile
import warnings
from laboneq._utils import id_generator
from laboneq.dsl.experiment import Section
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl import quantum
from laboneq.dsl.experiment import builtins_dsl
from laboneq.openqasm3.openqasm_error import OpenQasmException


class GateStore:
    """A mapping store between OpenQASM statements and LabOne Q.

    !!! version-changed "Deprecated in version 2.43.0"
        Use [OpenQASMTranspiler]() and [quantum.QuantumOperations]() instead.
    """

    def __init__(self):
        warnings.warn(
            "GateStore is deprecated. Use `OpenQASMTranspiler()` and `QuantumOperations()` instead.",
            FutureWarning,
            stacklevel=2,
        )
        self.gates: Dict[Tuple[str, Tuple[str, ...]], Callable[..., Section]] = {}
        self.gate_map: Dict[str, str] = {}
        self.waveforms: Dict[str, Callable] = {}
        self.ports = {}

    def lookup_gate(
        self,
        name: str,
        qubits: Tuple[str, ...],
        args: tuple | None = None,
        kwargs: dict | None = None,
    ) -> Section:
        kwargs = kwargs or {}
        args = args or ()
        return self.gates[(self.gate_map.get(name, name), qubits)](*args, **kwargs)

    def map_gate(self, qasm_name: str, labone_q_name: str):
        """Define mapping from qasm gate name to LabOne Q gate name."""
        self.gate_map[qasm_name] = labone_q_name

    def register_port(self, qasm_port: str, signal_line: str) -> None:
        self.ports[qasm_port] = signal_line

    def register_gate_section(
        self, name: str, qubit_names: list[str], section_factory: Callable[..., Section]
    ):
        """Register a LabOne Q section factory as a gate."""
        self.gates[(name, tuple(qubit_names))] = section_factory

    @staticmethod
    def _gate_pulse(qubit_uid, signal, pulse, phase, gate_id):
        def impl():
            gate = Section(uid=id_generator(f"{qubit_uid}_{gate_id}_pulse"))
            gate.play(
                signal=signal,
                pulse=pulse,
                increment_oscillator_phase=phase,
            )
            return gate

        return impl

    # TODO: cleaner interface & rename to register_drive_gate?
    def register_gate(
        self,
        name: str,
        qubit_name: str,
        pulse: Pulse | None,
        signal: str,
        phase=None,
        id=None,
    ):
        """Register a pulse as a single-qubit gate."""
        self.register_gate_section(
            name,
            (qubit_name,),
            self._gate_pulse(qubit_name, signal, pulse, phase, id),
        )

    def lookup_waveform(self, name: str) -> Pulse:
        return self.waveforms[name]

    def register_waveform(self, name: str, pulse: Pulse):
        self.waveforms[name] = pulse


class _GateStoreQuantumOperations(quantum.QuantumOperations):
    """QuantumOperations compatibility wrapper for GateStore."""

    QUBIT_TYPES = quantum.QuantumElement

    def __init__(
        self,
        gate_store: GateStore,
        qubits: dict[str, quantum.QuantumElement] | None = None,
    ):
        super().__init__()
        self._gate_store = gate_store
        self._qubit_map = {id(v): k for k, v in (qubits or {}).items()}
        available_gates = {op[0] for op in self._gate_store.gates}.union(
            {x for x in self._gate_store.gate_map}
        )
        for x in available_gates:
            self.register(self._create_gate(x), x)

    def _create_gate(self, name: str) -> Callable[..., Section]:
        def callback(qops, *args, **kwargs) -> Section:
            # Every quantum operation is called with qubits first as arguments (for now).
            qubits = tuple(
                [
                    self._qubit_map[id(q)]
                    for q in takewhile(
                        lambda x: isinstance(x, quantum.QuantumElement), args
                    )
                ]
            )
            args = tuple(args[len(qubits) :])
            try:
                section = self._gate_store.lookup_gate(name, qubits, args, kwargs)
            except KeyError as error:
                msg = f"Gate '{name}' for qubit(s) {qubits} not found."
                raise OpenQasmException(msg) from error
            if not isinstance(section, Section):
                raise OpenQasmException("GateStore gates must return 'Section'")
            # Replace the section created by the quantum operation machinery with the one returned by the gate
            qops_section = builtins_dsl.active_section()
            qops_section.uid = section.uid
            for child in section.children:
                qops_section.add(child)

        return callback
