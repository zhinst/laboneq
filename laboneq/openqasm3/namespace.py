# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from laboneq.openqasm3.openqasm_error import OpenQasmException


@dataclass
class QubitRef:
    canonical_name: str


@dataclass
class ClassicalRef:
    """A reference to a classical value"""

    value: Any
    canonical_name: str


@dataclass
class Frame:
    """A frame data structure"""

    canonical_name: str
    port: str
    frequency: float
    phase: float


class Namespace:
    def __init__(self, toplevel: bool | None = True):
        self.toplevel = toplevel
        self.local_scope: dict[str, Any] = {}

    def declare_qubit(self, name: str) -> QubitRef:
        if self.toplevel is False:
            raise OpenQasmException("Qubit declaration is illegal in this scope")

        if name not in self.local_scope:
            self.local_scope[name] = QubitRef(name)
        else:
            msg = f"Name '{name}' already exists"
            raise OpenQasmException(msg)
        return self.local_scope[name]

    def declare_classical_value(
        self, name: str, value: Any, const: bool = False
    ) -> ClassicalRef | list[ClassicalRef]:
        # For now, classical values are not resources which we 'allocate'.
        # Instead we treat them as references (to Python objects).
        # TODO: Enforce constantness of constants. Currently they are implemented as variables
        return self.declare_reference(name, value)

    def declare_frame(
        self, name: str, port: str, frequency: float, phase: float
    ) -> Frame:
        if name not in self.local_scope:
            if phase != 0:
                msg = "Nonzero frame phase is not supported yet"
                raise OpenQasmException(msg)
            self.local_scope[name] = Frame(name, port, frequency, phase)
        else:
            msg = f"Name '{name}' already exists"
            raise OpenQasmException(msg)
        return self.local_scope[name]

    def declare_reference(
        self, name: str, value: Any = None
    ) -> ClassicalRef | list[ClassicalRef]:
        if name not in self.local_scope:
            if isinstance(value, (QubitRef, ClassicalRef, list)):
                # These do not need another layer of indirection
                self.local_scope[name] = value
            else:
                self.local_scope[name] = ClassicalRef(value, name)
        else:
            msg = f"Name '{name}' already exists"
            raise OpenQasmException(msg)
        return self.local_scope[name]


class NamespaceNest:
    """A stack of namespaces, with the the current namespace on top.

    The attribute `current` can be used to add variables to the deepest nesting.
    """

    def __init__(self):
        self._nesting = [Namespace(toplevel=True)]

    def open(self) -> None:
        self._nesting.append(Namespace(toplevel=False))

    def close(self) -> None:
        self._nesting.pop()

    @property
    def current(self) -> Namespace:
        return self._nesting[-1]

    def lookup(
        self, name: str
    ) -> QubitRef | ClassicalRef | list[QubitRef] | list[ClassicalRef]:
        for namespace in reversed(self._nesting):
            if name in namespace.local_scope:
                return namespace.local_scope[name]
        else:
            raise KeyError(name)
