# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""This module defines the QuantumPlatform and QPU classes.

A `QPU` contains the "physics" of a quantum device -- the quantum element parameters
and definition of operations on quantum elements.

A `QuantumPlatform` contains the `QPU`, and the `DeviceSetup` which describes
the control hardware used to interface to the device.

By itself a `QPU` provides everything needed to *build* or *design* an
experiment for a quantum device. The `DeviceSetup` provides the additional
information needed to *compile* an experiment for specific control hardware.

Together these provide a `QuantumPlatform` -- i.e. everything needed to build,
compile and run experiments on real devices.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, Sequence
from typing_extensions import deprecated

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter
from laboneq.dsl.quantum.qpu_topology import QPUTopology
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.dsl.session import Session
from laboneq.dsl.quantum.quantum_operations import QuantumOperations

if TYPE_CHECKING:
    from laboneq.dsl.device import DeviceSetup
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
        quantum_elements = ", ".join(q.uid for q in self.qpu.quantum_elements)
        return (
            f"<{type(self).__qualname__}"
            f" setup={self.setup.uid!r}"
            f" qpu.quantum_elements=[{quantum_elements}]"
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
                to an emulator (in the case of 'True'),
                or the real system (in the case of 'False')
        """
        session = Session(self.setup)
        session.connect(do_emulation=do_emulation)
        return session


@classformatter
class QPU:
    """A Quantum Processing Unit (QPU).

    A `QPU` provides the logical description of a quantum device needed to *build*
    experiments for it. For example, the quantum element parameters and the definition of
    operations on those quantum elements.

    It does not provide a description of the control hardware needed to *compile* an
    experiment.

    In short, a `QPU` defines the device physics and a `DeviceSetup` defines the control
    hardware being used.

    !!! version-changed "Deprecated in version 2.52.0."
        The argument `qubits` was deprecated and replaced with the argument `quantum_elements`.

    Arguments:
        quantum_elements:
            The quantum elements to run the experiments on.
        quantum_operations:
            The quantum operations to use when building the experiment.

    Attributes:
        topology:
            The topology information for the QPU.
    """

    def __init__(
        self,
        quantum_elements: QuantumElements,
        quantum_operations: QuantumOperations | type[QuantumOperations],
    ) -> None:
        self.quantum_elements: list[QuantumElement] = (
            [quantum_elements]
            if isinstance(quantum_elements, QuantumElement)
            else list(quantum_elements)
        )
        self._quantum_element_map = {q.uid: q for q in self.quantum_elements}

        if isinstance(quantum_operations, QuantumOperations):
            quantum_operations.attach_qpu(self)
        elif inspect.isclass(quantum_operations) and issubclass(
            quantum_operations, QuantumOperations
        ):
            quantum_operations = quantum_operations(self)
        else:
            raise TypeError(
                f"quantum_operations has invalid type: {type(quantum_operations)}. "
                f"Expected type: QuantumOperations | type[QuantumOperations]."
            )

        self.quantum_operations = quantum_operations
        self.topology = QPUTopology(self.quantum_elements)

    def __repr__(self) -> str:
        quantum_elements = ", ".join(q.uid for q in self.quantum_elements)
        return (
            f"<{type(self).__qualname__}"
            f" quantum_elements=[{quantum_elements}]"
            f" quantum_operations={type(self.quantum_operations).__qualname__}"
            f" topology={self.topology}"
            f">"
        )

    def __rich_repr__(self):
        yield "quantum_elements", [q.uid for q in self.quantum_elements]
        yield "quantum_operations", type(self.quantum_operations).__qualname__
        yield "topology", self.topology

    def copy_quantum_elements(self) -> QuantumElements:
        """Return new quantum elements that are a copy of the original quantum elements."""
        return [q.replace() for q in self.quantum_elements]

    @deprecated(
        "The .copy_qubits method is deprecated. Use `.copy_quantum_elements` instead.",
        category=FutureWarning,
    )
    def copy_qubits(self) -> QuantumElements:
        """Return new qubits that are a copy of the original qubits.

        !!! version-changed "Deprecated in version 2.52.0."
            The method `copy_qubits` was deprecated and replaced with the method `copy_quantum_elements`.
        """
        return self.copy_quantum_elements()

    @classmethod
    def _get_invalid_param_paths(
        cls, quantum_element, overrides: dict[str, Any]
    ) -> Sequence:
        invalid_params = []
        for param_path in overrides:
            keys = param_path.split(".")
            obj = quantum_element.parameters
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

    def override_quantum_elements(
        self,
        quantum_element_parameters: dict[
            str, dict[str, int | float | str | dict | None]
        ],
    ) -> QPU:
        """Override quantum element parameters and return a new QPU.

        !!! note
            This method detaches the quantum operations from the QPU and attaches them to the new QPU.

        Arguments:
            quantum_element_parameters:
                The quantum elements and their parameters that need to be updated passed a dict
                of the form:
                    ```python
                    {qb_uid: {qb_param_name: qb_param_value}}
                    ```
        Returns:
            A new QPU with overridden quantum element parameters.
        Raises:
            ValueError:
                If one of the quantum elements passed is not found in the qpu.
                If one of the parameters passed is not found in the quantum element.
        """
        self.quantum_operations.detach_qpu()
        new_qpu = QPU(self.copy_quantum_elements(), self.quantum_operations)
        new_qpu.update_quantum_elements(quantum_element_parameters)
        return new_qpu

    @deprecated(
        "The .override_qubits method is deprecated. Use `.override_quantum_elements` instead.",
        category=FutureWarning,
    )
    def override_qubits(
        self, qubit_parameters: dict[str, dict[str, int | float | str | dict | None]]
    ) -> QPU:
        """Override qubit parameters and return a new QPU.

        !!! note
            This method detaches the quantum operations from the QPU and attaches them to the new QPU.

        !!! version-changed "Deprecated in version 2.52.0."
            The method `override_qubits` was deprecated and replaced with the method `override_quantum_elements`.

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
        return self.override_quantum_elements(qubit_parameters)

    def update_quantum_elements(
        self,
        quantum_element_parameters: dict[
            str, dict[str, int | float | str | dict | None]
        ],
    ) -> None:
        """Updates quantum element parameters.

        Arguments:
            quantum_element_parameters:
                The quantum elements and their parameters that need to be updated passed a dict
                of the form:
                    ```python
                    {qb_uid: {qb_param_name: qb_param_value}}
                    ```
        Raises:
            ValueError:
                If one of the quantum elements passed is not found in the qpu.
                If one of the parameters passed is not found in the quantum element.
        """
        invalid_params = []
        for qid, params_dict in quantum_element_parameters.items():
            if qid not in self._quantum_element_map:
                raise ValueError(f"Quantum element {qid} was not found in the QPU.")
            quantum_element = self._quantum_element_map[qid]
            invalid_params += self._get_invalid_param_paths(
                quantum_element, params_dict
            )
        if invalid_params:
            raise ValueError(
                f"Update parameters do not match the quantum element "
                f"parameters: {invalid_params}.",
            )

        for qid, params_dict in quantum_element_parameters.items():
            self._quantum_element_map[qid].update(**params_dict)

    @deprecated(
        "The .update_qubits method is deprecated. Use `.update_quantum_elements` instead.",
        category=FutureWarning,
    )
    def update_qubits(
        self,
        qubit_parameters: dict[str, dict[str, int | float | str | dict | None]],
    ) -> None:
        """Updates qubit parameters.

        !!! version-changed "Deprecated in version 2.52.0."
            The method `update_qubits` was deprecated and replaced with the method `update_quantum_elements`.

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
        return self.update_quantum_elements(qubit_parameters)

    @staticmethod
    def measure_section_length(quantum_elements: QuantumElements) -> float:
        """Calculates the length of the measure section for multiplexed readout.

        In order to allow the quantum elements to have different readout and/or integration
        lengths, the measure section length needs to be fixed to the longest one
        across the quantum elements used in the experiment.

        Args:
            quantum_elements:
                The quantum elements that are being measured.
        Returns:
            The length of the multiplexed-readout measure section.
        """
        # TODO: Only works on the TunableTransmonQubit from laboneq_applications
        #       currently.
        return max(
            [q.readout_integration_parameters()[1]["length"] for q in quantum_elements]
        )

    def quantum_element_by_uid(self, uid: str) -> QuantumElement:
        """Returns quantum element by UID.

        Arguments:
            uid: Unique identifier of the quantum element within the QPU.
        Returns:
            Quantum element with given `uid`.
        Raises:
            KeyError: Quantum element does not exist.
        """
        try:
            return self._quantum_element_map[uid]
        except KeyError:
            msg = f"Quantum element {uid} does not exist in the QPU."
            raise KeyError(msg) from None

    @deprecated(
        "The .qubit_by_uid method is deprecated. Use `.quantum_element_by_uid` instead.",
        category=FutureWarning,
    )
    def qubit_by_uid(self, uid: str) -> QuantumElement:
        """Returns qubit by UID.

        !!! version-changed "Deprecated in version 2.52.0."
            The method `qubit_by_uid` was deprecated and replaced with the method `quantum_element_by_uid`.

        Arguments:
            uid: Unique identifier of the qubit within the QPU.
        Returns:
            Qubit with given `uid`.
        Raises:
            KeyError: Qubit does not exist.
        """
        return self.quantum_element_by_uid(uid)

    @property
    @deprecated(
        "The .qubits attribute is deprecated. Use `.quantum_elements` instead.",
        category=FutureWarning,
    )
    def qubits(self) -> QuantumElements:
        """Return qubits.

        !!! version-changed "Deprecated in version 2.52.0."
            The attribute `qubits` was deprecated and replaced with the attribute `quantum_elements`.
        """
        return self.quantum_elements

    @property
    @deprecated(
        "The ._qubit_map attribute is deprecated. Use `._quantum_element_map` instead.",
        category=FutureWarning,
    )
    def _qubit_map(self) -> dict[str, QuantumElement]:
        """Return qubit map.

        !!! version-changed "Deprecated in version 2.52.0."
            The attribute `_qubit_map` was deprecated and replaced with the attribute `_quantum_element_map`.
        """
        return self._quantum_element_map
