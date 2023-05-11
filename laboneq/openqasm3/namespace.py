# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from laboneq.openqasm3.openqasm_error import OpenQasmException


@dataclass
class QubitRef:
    canonical_name: str


@dataclass
class ClassicalRef:
    """A reference to a classical value"""

    value: Any
    canonical_name: str


class Namespace:
    def __init__(self, parent: Optional = None):
        self.parent = parent
        self.local_scope = {}

    def declare_qubit(self, name: str) -> QubitRef:
        if self.parent is not None:
            raise OpenQasmException("Qubit declaration is illegal in this scope")

        if name not in self.local_scope:
            self.local_scope[name] = QubitRef(name)
        else:
            raise OpenQasmException(f"Name '{name}' already exists")
        return self.local_scope[name]

    def declare_classical_value(self, name: str, value: Any) -> ClassicalRef:
        # For now, classical values are not resources which we 'allocate'.
        # Instead we treat them as references (to Python objects).
        return self.declare_reference(name, value)

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
            raise OpenQasmException(f"Name '{name}' already exists")
        return self.local_scope[name]

    def lookup(
        self, name: str
    ) -> QubitRef | ClassicalRef | list[ClassicalRef] | list[QubitRef]:
        if name in self.local_scope:
            return self.local_scope[name]
        elif self.parent is not None:
            return self.parent.lookup(name)
        else:
            raise KeyError(name)
