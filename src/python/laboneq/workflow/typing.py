# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Common type hints for laboneq.workflows.

This module provides a set of common types for use within the
laboneq.workflow package.

Type hints
----------

* [Qubits]()

    Either a single qubit or a sequence of qubits.

* [SimpleDict]()

    Simple dictionaries are used for artifact metadata and serializer
    options, allowing these to themselves be serialized easily, especially
    to JSON.

    Simple dictionaries only allow Python strings as keys and their values
    may be simple numeric types, strings or `None`.
"""

from __future__ import annotations

from typing import TypeAlias


from laboneq.dsl.quantum import QuantumElement
from collections.abc import Sequence

__all__ = [
    "QuantumElements",
    "SimpleDict",
]

SimpleDict: TypeAlias = dict[str, str | int | float | complex | bool | None]
QuantumElements: TypeAlias = QuantumElement | Sequence[QuantumElement]
