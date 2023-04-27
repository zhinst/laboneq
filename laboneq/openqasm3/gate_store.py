# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional, Tuple

from laboneq.dsl.experiment import Section
from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.utils import id_generator


class GateStore:
    def __init__(self):
        self.gates: Dict[Tuple[str, Tuple[int, ...]], Callable[..., Section]] = {}
        self.gate_map: Dict[str, str] = {}
        self.qubit_lsg_map: Dict[int, str] = {}

    def lookup_gate(self, name: str, qubits: Tuple[int, ...], args=()) -> Section:
        return self.gates[(self.gate_map.get(name, name), qubits)](*args)

    def map_gate(self, qasm_name: str, l1q_name: str):
        """Define mapping from qasm gate name to L1Q gate name."""
        self.gate_map[qasm_name] = l1q_name

    def register_gate_section(self, name, qubits, section_factory):
        """Register a LabOne Q section factory as a gate."""
        self.gates[(name, qubits)] = section_factory

    @staticmethod
    def _gate_pulse(qubit: str, pulse, phase, gate_id):
        def impl():
            gate = Section(uid=id_generator(f"{qubit}_{gate_id}_pulse"))
            gate.play(
                signal=f"{qubit}_drive",
                pulse=pulse,
                increment_oscillator_phase=phase,
            )
            return gate

        return impl

    def map_qubits(self, map: Dict[int, str]):
        """Define mapping from qasm qubit names to LabOne Q logical signal groups.

        Required by `GateStore.register_gate()`.
        """
        self.qubit_lsg_map.update(map)

    def register_gate(
        self, name: str, qubit: int, pulse: Optional[Pulse], phase=None, id=None
    ):
        """Register a pulse as a single-qubit gate."""
        self.register_gate_section(
            name,
            (qubit,),
            self._gate_pulse(self.qubit_lsg_map[qubit], pulse, phase, id),
        )