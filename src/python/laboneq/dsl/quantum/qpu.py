# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the QuantumPlatform and QPU classes.

A `QPU` contains the "physics" of a quantum device -- the qubit parameters
and definition of operations on qubits.

A `QuantumPlatform` contains the `QPU`, and the `DeviceSetup` which describes
the control hardware used to interface to the device.

By itself a `QPU` provides everything needed to *build* or *design* an
experiment for a quantum device. The `DeviceSetup` provides the additional
information needed to *compile* an experiment for specific control hardware.

Together these provide a `QuantumPlatform` -- i.e. everything needed to build,
compile and run experiments on real devices.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.session import Session

if TYPE_CHECKING:
    from laboneq.dsl.device import DeviceSetup
    from laboneq.dsl.quantum.quantum_operations import QuantumOperations
    from laboneq.workflow.typing import QuantumElements


@classformatter
class QuantumPlatform:
    """A quantum hardware platform.

    A `QuantumPlatform` provides the logical description of a quantum device needed to
    define experiments (the `QPU`) and the description of the control hardware needed to
    compile an experiment (the `DeviceSetup`).

    In short, a `QPU` defines the device physics and a `DeviceSetup` defines the control
    hardware being used.

    Arguments:
        setup:
            The `DeviceSetup` describing the control hardware of the device.
        qpu:
            The `QPU` describing the parameters and topology of the quantum device
            and providing the definition of quantum operations on the device.
    """

    def __init__(
        self,
        setup: DeviceSetup,
        qpu: QPU,
    ) -> None:
        """Initialize a new QPU.

        Arguments:
            setup:
                The device setup to use when running an experiment.
            qpu:
                The QPU to use when building an experiment.
        """
        self.setup = setup
        self.qpu = qpu

    def __repr__(self) -> str:
        qubits = ", ".join(q.uid for q in self.qpu.qubits)
        return (
            f"<{type(self).__qualname__}"
            f" setup={self.setup.uid!r}"
            f" qpu.qubits=[{qubits}]"
            f" qpu.quantum_operations={type(self.qpu.quantum_operations).__qualname__}"
            f">"
        )

    def __rich_repr__(self):
        yield "setup", self.setup.uid
        yield "qpu", self.qpu

    def session(self, do_emulation: bool = False) -> Session:  # noqa: FBT001 FBT002
        """Return a new LabOne Q session.

        Arguments:
            do_emulation:
                Specifies if the session should connect
                to a emulator (in the case of 'True'),
                or the real system (in the case of 'False')
        """
        session = Session(self.setup)
        session.connect(do_emulation=do_emulation)
        return session


@classformatter
class QPU:
    """A Quantum Processing Unit (QPU).

    A `QPU` provides the logical description of a quantum device needed to *build*
    experiments for it. For example, the qubit parameters and the definition of
    operations on those qubits.

    It does not provide a description of the control hardware needed to *compile* an
    experiment.

    In short, a `QPU` defines the device physics and a `DeviceSetup` defines the control
    hardware being used.

    Arguments:
        qubits:
            The qubits to run the experiments on.
        quantum_operations:
            The quantum operations to use when building the experiment.
    """

    def __init__(
        self,
        qubits: QuantumElements,
        quantum_operations: QuantumOperations,
    ) -> None:
        self.qubits: list[QuantumElement] = (
            [qubits] if isinstance(qubits, QuantumElement) else list(qubits)
        )
        self.quantum_operations = quantum_operations
        self._qubit_map = {q.uid: q for q in qubits}

    def __repr__(self) -> str:
        qubits = ", ".join(q.uid for q in self.qubits)
        return (
            f"<{type(self).__qualname__}"
            f" qubits=[{qubits}]"
            f" quantum_operations={type(self.quantum_operations).__qualname__}"
            f">"
        )

    def __rich_repr__(self):
        yield "qubits", [q.uid for q in self.qubits]
        yield "quantum_operations", type(self.quantum_operations).__qualname__

    def copy_qubits(self) -> QuantumElements:
        """Return new qubits that are a copy of the original qubits."""
        return [q.replace() for q in self.qubits]

    @classmethod
    def _get_invalid_param_paths(cls, qubit, overrides: dict[str, Any]) -> Sequence:
        invalid_params = []
        for param_path in overrides:
            keys = param_path.split(".")
            obj = qubit.parameters
            for key in keys:
                if isinstance(obj, dict):
                    if key not in obj:
                        invalid_params.append(param_path)
                        break
                    obj = obj[key]
                elif not hasattr(obj, key):
                    invalid_params.append(param_path)
                    break
                else:
                    obj = getattr(obj, key)
        return invalid_params

    def override_qubits(
        self, qubit_parameters: dict[str, dict[str, int | float | str | dict | None]]
    ) -> QPU:
        """Override qubit parameters and return a new QPU.

        Arguments:
            qubit_parameters:
                The qubits and their parameters that need to be updated passed a dict
                of the form:
                    ```python
                    {qb_uid: {qb_param_name: qb_param_value}}
                    ```
        Returns:
            A new QPU with overridden qubit parameters.
        Raises:
            ValueError:
                If one of the qubits passed is not found in the qpu.
                If one of the parameters passed is not found in the qubit.
        """
        new_qpu = QPU(self.copy_qubits(), self.quantum_operations)
        new_qpu.update_qubits(qubit_parameters)
        return new_qpu

    def update_qubits(
        self,
        qubit_parameters: dict[str, dict[str, int | float | str | dict | None]],
    ) -> None:
        """Updates qubit parameters.

        Arguments:
            qubit_parameters:
                The qubits and their parameters that need to be updated passed a dict
                of the form:
                    ```python
                    {qb_uid: {qb_param_name: qb_param_value}}
                    ```

        Raises:
            ValueError:
                If one of the qubits passed is not found in the qpu.
                If one of the parameters passed is not found in the qubit.
        """
        invalid_params = []
        for qid, params_dict in qubit_parameters.items():
            if qid not in self._qubit_map:
                raise ValueError(f"Qubit {qid} was not found in the QPU.")
            qubit = self._qubit_map[qid]
            invalid_params += self._get_invalid_param_paths(qubit, params_dict)
        if invalid_params:
            raise ValueError(
                f"Update parameters do not match the qubit "
                f"parameters: {invalid_params}.",
            )

        for qid, params_dict in qubit_parameters.items():
            self._qubit_map[qid].update(**params_dict)

    @staticmethod
    def measure_section_length(qubits: QuantumElements) -> float:
        """Calculates the length of the measure section for multiplexed readout.

        In order to allow the qubits to have different readout and/or integration
        lengths, the measure section length needs to be fixed to the longest one
        across the qubits used in the experiment.

        Args:
            qubits:
                The qubits that are being measured.

        Returns:
            The length of the multiplexed-readout measure section.
        """
        # TODO: Only works on the TunableTransmonQubit from laboneq_applications
        #       currently.
        return max([q.readout_integration_parameters()[1]["length"] for q in qubits])

    def qubit_by_uid(self, uid: str) -> QuantumElement:
        """Returns qubit by UID.

        Arguments:
            uid: Unique identifier of the qubit within the QPU.

        Returns:
            Qubit with given `uid`.

        Raises:
            KeyError: Qubit does not exist.
        """
        try:
            return self._qubit_map[uid]
        except KeyError:
            msg = f"Qubit {uid} does not exist in the QPU."
            raise KeyError(msg) from None
