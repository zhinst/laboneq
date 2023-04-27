# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class ArrayVariable:
    size: int
    qubit_start_index: int


@dataclass
class Variable:
    qubit_index: int


@dataclass
class VariableInArray:
    array: ArrayVariable
    array_index: int


@dataclass
class VariableStore:
    variables: Dict[str, Union[ArrayVariable, Variable, VariableInArray]] = field(
        default_factory=dict
    )
    current_index = 0

    def add_array_variable(self, name, size):
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists.")
        self.variables[name] = ArrayVariable(size, self.current_index)
        self.current_index += size

    def add_variable(self, name):
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists.")
        self.variables[name] = Variable(self.current_index)
        self.current_index += 1

    def add_alias(self, name, target, index=None):
        # todo: Alias multiple qubits of an array to a new array
        if name in self.variables:
            raise ValueError(f"Variable '{name}' already exists.")
        try:
            if index is None:
                self.variables[name] = self.variables[target]
            else:
                target_var = self.variables[target]
                if not isinstance(target_var, ArrayVariable):
                    if index == 0:
                        self.variables[name] = self.variables[target]
                        return
                    else:
                        raise ValueError(f"Variable '{target}' is not an array.")
                if index >= target_var.size:
                    raise ValueError(f"Index {index} out of range for array {target}.")
                self.variables[name] = VariableInArray(target_var, index)
        except KeyError:
            raise ValueError(f"Alias target '{target}' not found.")

    def get_qubit_number(self, name, index=None):
        try:
            variable = self.variables[name]
        except KeyError as e:
            raise KeyError(f"Variable '{name}' not found.") from e
        if isinstance(variable, Variable):
            if index is not None and index > 0:
                raise ValueError(f"Variable '{name}' is not an array.")
            qubit = variable.qubit_index
        elif isinstance(variable, VariableInArray):
            if index is not None:
                raise ValueError(f"Variable '{name}' is not an array.")
            qubit = variable.array.qubit_start_index + variable.array_index
        else:
            assert isinstance(variable, ArrayVariable)
            if index is None:
                raise ValueError(f"Index required for array variable {name}.")
            if index >= variable.size:
                raise ValueError(f"Index {index} out of range for array {name}.")
            qubit = variable.qubit_start_index + index
        return qubit
