# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Optional, Tuple

from laboneq._utils import id_generator
from laboneq.dsl.experiment import Section
from laboneq.dsl.experiment.pulse import Pulse


class GateStore:
    def __init__(self):
        self.gates: Dict[Tuple[str, Tuple[str, ...]], Callable[..., Section]] = {}
        self.gate_map: Dict[str, str] = {}

    def lookup_gate(
        self, name: str, qubits: Tuple[str, ...], args=(), kwargs=None
    ) -> Section:
        kwargs = kwargs or {}
        return self.gates[(self.gate_map.get(name, name), qubits)](*args, **kwargs)

    def map_gate(self, qasm_name: str, labone_q_name: str):
        """Define mapping from qasm gate name to LabOne Q gate name."""
        self.gate_map[qasm_name] = labone_q_name

    def register_gate_section(self, name: str, qubit_names: list[str], section_factory):
        """Register a LabOne Q section factory as a gate."""
        self.gates[(name, qubit_names)] = section_factory

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
        pulse: Optional[Pulse],
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
