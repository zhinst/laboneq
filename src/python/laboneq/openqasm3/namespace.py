# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import abc
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable
from laboneq.dsl.experiment.pulse import Pulse, PulseSampled
from laboneq.openqasm3.openqasm_error import OpenQasmException


@dataclass
class QubitRef:
    """A reference to a qubit"""

    canonical_name: str


@dataclass
class ClassicalRef:
    """A reference to a classical value"""

    canonical_name: str
    value: Any


@dataclass
class Frame:
    """A frame data structure"""

    canonical_name: str
    port: str
    frequency: float
    phase: float


@dataclass
class Function:
    """A callable function"""

    canonical_name: str
    func: Callable
    # TODO: Add argument types
    # TODO: Add return type


@dataclass
class Waveform(ClassicalRef):
    """A waveform data"""

    canonical_name: str
    value: list | Pulse

    def __post_init__(self):
        if not isinstance(self.value, (list, Pulse)):
            msg = "Waveform must be a list of samples or a LabOneQ Pulse object."
            raise OpenQasmException(msg)

    def pulse(self) -> Pulse:
        if isinstance(self.value, Pulse):
            return self.value
        else:
            return PulseSampled(samples=self.value)


@dataclass
class Port(ClassicalRef):
    """An abstract port.

    Attributes:
        canonical_name: Declared name.
        qubit: Qubit UID.
        value: Signal the port is attached to.
    """

    canonical_name: str
    qubit: str
    value: str


@dataclass
class QubitRegisterRef:
    """A register of qubits"""

    canonical_name: str
    value: list[QubitRef]

    def __len__(self):
        return len(self.value)


@dataclass
class Array:
    """An array of classical values"""

    canonical_name: str
    value: list

    def __len__(self):
        return len(self.value)


class BaseNamespace(abc.ABC):
    @abc.abstractmethod
    def declare_qubit(self, name): ...

    @abc.abstractmethod
    def declare_classical_value(self, name, value): ...

    @abc.abstractmethod
    def declare_port(self, name: str, qubit: str, value: str) -> Port:
        """Declare abstract port."""

    @abc.abstractmethod
    def declare_frame(
        self, name: str, port: str, frequency: float, phase: float
    ) -> Frame:
        """Declare abstract frame."""

    @abc.abstractmethod
    def declare_waveform(self, name: str, value: Any) -> Waveform:
        """Declare abstract waveform."""

    @abc.abstractmethod
    def declare_reference(self, name, value): ...

    @abc.abstractmethod
    def declare_function(self, name, value): ...


_NULL = object()


class DefaultNamespace(BaseNamespace):
    """
    Default namespace implementation.

    Attributes:
        _VISIBLE_VARIABLES (frozenset):
            A set of allowed variable types that can be retrieved by this namespace.
        local_scope (dict):
            A dictionary that maps variable names to their corresponding references.
    """

    _VISIBLE_VARIABLES = frozenset(
        [Array, ClassicalRef, Frame, Function, QubitRef, QubitRegisterRef, Waveform]
    )

    def __init__(self):
        self.local_scope: dict[
            str, ClassicalRef | QubitRef | Array | QubitRegisterRef | Function
        ] = {}

    def declare_qubit(self, name: str) -> QubitRef:
        self._check_duplicate(name)
        self.local_scope[name] = QubitRef(name)
        return self.local_scope[name]

    def declare_classical_value(
        self, name: str, value: Any, const: bool = False
    ) -> ClassicalRef | Array:
        # For now, classical values are not resources which we 'allocate'.
        # Instead we treat them as references (to Python objects).
        # TODO: Enforce constantness of constants. Currently they are implemented as variables
        return self.declare_reference(name, value)

    def declare_port(self, name: str, qubit: str, value: str) -> Port:
        self._check_duplicate(name)
        self.local_scope[name] = Port(canonical_name=name, qubit=qubit, value=value)
        return self.local_scope[name]

    def declare_frame(
        self, name: str, port: str, frequency: float, phase: float
    ) -> Frame:
        self._check_duplicate(name)
        if phase != 0:
            msg = "Nonzero frame phase is not supported yet"
            raise OpenQasmException(msg)
        self.local_scope[name] = Frame(name, port, frequency, phase)
        return self.local_scope[name]

    def declare_reference(
        self, name: str, value: Any = None
    ) -> ClassicalRef | Array | QubitRef | QubitRegisterRef:
        self._check_duplicate(name)
        if isinstance(value, (QubitRef, ClassicalRef, QubitRegisterRef, Array)):
            # These do not need another layer of indirection
            self.local_scope[name] = value
        elif isinstance(value, list):
            # simple type check
            # TODO: Find a more robust way to add reference to namespace
            if isinstance(value[0], QubitRef):
                self.local_scope[name] = QubitRegisterRef(name, value)
            else:
                self.local_scope[name] = Array(name, value)
        else:
            self.local_scope[name] = ClassicalRef(name, value)
        return self.local_scope[name]

    def declare_function(
        self,
        name: str,
        arguments,
        return_type,
        func: Callable,
    ) -> Function:
        # TODO: Check what to do if already defined
        if name not in self.local_scope:
            # TODO: Store argument and return types
            self.local_scope[name] = Function(canonical_name=name, func=func)

        return self.local_scope[name]

    def declare_waveform(self, name: str, value: Any) -> Waveform:
        self._check_duplicate(name)
        if not isinstance(value, (list, Pulse)):
            msg = "Waveform must be a list of samples or a LabOneQ Pulse object."
            raise OpenQasmException(msg)
        self.local_scope[name] = Waveform(name, value)
        return self.local_scope[name]

    def _check_duplicate(self, name):
        if name in self.local_scope:
            msg = f"Name '{name}' already exists"
            raise OpenQasmException(msg)

    def lookup(
        self, name: str, namespace: BaseNamespace | None = None
    ) -> QubitRef | ClassicalRef | Array | QubitRegisterRef | Function:
        ret = self.local_scope.get(name, _NULL)
        if namespace is not None and hasattr(namespace, "_VISIBLE_VARIABLES"):
            if not isinstance(ret, tuple(namespace._VISIBLE_VARIABLES)):
                return _NULL
        return ret


class TopLevelNamespace(DefaultNamespace): ...


class LocalBlockNamespace(DefaultNamespace):
    def declare_qubit(self, name: str) -> QubitRef:
        raise OpenQasmException(
            "Qubit declaration is illegal outside of the top-level scope"
        )


class NamespaceStack:
    """A stack of namespaces, with the the current namespace on top.

    The attribute `current` can be used to add variables to the deepest nesting.
    """

    def __init__(self):
        self._nesting = [TopLevelNamespace()]

    def open(self, namespace_type: BaseNamespace = LocalBlockNamespace) -> None:
        # TODO: Add support for other scope types: Gate/Function scope, CalibrationScope
        self._nesting.append(namespace_type())

    def close(self) -> None:
        self._nesting.pop()

    @property
    def current(self) -> BaseNamespace:
        return self._nesting[-1]

    def lookup(
        self, name: str
    ) -> QubitRef | ClassicalRef | Array | QubitRegisterRef | Function:
        for namespace in reversed(self._nesting):
            ret = namespace.lookup(name, namespace=self.current)
            if ret != _NULL:
                return ret
        else:
            raise KeyError(name)

    @contextmanager
    def new_scope(self, namespace_type: BaseNamespace = LocalBlockNamespace):
        self.open(namespace_type)
        yield
        self.close()
