# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, TextIO, Tuple, Union

import openpulse
import openpulse.ast as ast

import openqasm3.visitor
from laboneq.dsl.experiment import Section

ALLOWED_NODE_TYPES = {
    ast.Program,
    ast.QuantumGate,
    ast.Identifier,
    ast.Span,
    ast.QubitDeclaration,
    ast.AliasStatement,
    ast.IndexExpression,
    ast.Include,
    ast.IntegerLiteral,
    ast.RangeDefinition,
    ast.IndexedIdentifier,
    ast.QuantumReset,
}


class AllowedNodeTypesVisitor(openqasm3.visitor.QASMVisitor):
    def generic_visit(self, node: ast.QASMNode, context=None):
        if type(node) not in ALLOWED_NODE_TYPES:
            raise TypeError(f"Node type {type(node)} not yet supported")
        super().generic_visit(node, context)


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
    qubit_map: Dict[int, int] = field(default_factory=dict)
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
        return self.qubit_map.get(qubit, qubit)


def get_collection_and_single_index(
    expression: Union[ast.IndexExpression, ast.IndexedIdentifier]
):
    # todo: DiscreteSet as index
    if isinstance(expression, ast.IndexExpression):
        name_identifier = expression.collection
        if len(expression.index) != 1:
            return "invalid", None
        ex0 = expression.index[0]
    else:
        assert isinstance(expression, ast.IndexedIdentifier)
        name_identifier = expression.name
        if len(expression.indices) != 1 or len(expression.indices[0]) != 1:
            return "invalid", None
        ex0 = expression.indices[0][0]

    if not isinstance(name_identifier, ast.Identifier):
        return None, None
    collection = name_identifier.name
    if isinstance(ex0, ast.RangeDefinition):
        if not (
            isinstance(ex0.start, ast.IntegerLiteral)
            and isinstance(ex0.end, ast.IntegerLiteral)
            and ex0.start.value == ex0.end.value
        ):
            return collection, None
        return collection, ex0.start.value
    elif isinstance(ex0, ast.IntegerLiteral):
        return collection, ex0.value
    else:
        return collection, None


class OpenQasm3Importer:
    def __init__(
        self,
        gate_store: Dict[Tuple[str, Tuple[int, ...]], Section],
        *,
        gate_map: Optional[Dict[str, str]] = None,
        qubit_map: Optional[Dict[int, int]] = None,
    ):
        self.gate_store = gate_store
        self.gate_map = gate_map or {}
        self.variables = VariableStore(qubit_map or {})

    def __call__(
        self,
        text: Optional[str] = None,
        file: Optional[TextIO] = None,
        filename: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> Section:
        if [bool(arg) for arg in [text, file, filename, stream]].count(True) != 1:
            raise ValueError(
                "Must specify exactly one of text, file, filename, or stream"
            )
        if filename:
            with open(filename, "r") as f:
                return self._import_text(f.read())
        elif file:
            return self._import_text(file.read())
        elif stream:
            return self._import_text(stream.read())
        else:
            return self._import_text(text)

    def _import_text(self, text) -> Section:
        tree = openpulse.parse(text)
        assert isinstance(tree, ast.Program)
        AllowedNodeTypesVisitor().visit(tree, None)

        root = Section()
        try:
            for statement in tree.statements:
                if isinstance(statement, ast.QubitDeclaration):
                    self._handle_qubit_declaration(statement)
                elif isinstance(statement, ast.AliasStatement):
                    self._handle_alias_statement(statement)
                elif isinstance(statement, ast.Include):
                    self._handle_include(statement)
                elif isinstance(statement, ast.QuantumGate):
                    self._handle_quantum_gate(statement, root)
                elif isinstance(statement, ast.QuantumReset):
                    self._handle_quantum_reset(statement, root)
                else:
                    raise ValueError(f"Statement type {type(statement)} not supported")

            return root
        except Exception as e:
            if sys.version_info >= (3, 11):
                e.add_note(
                    "See line(s) {statement.span.start_line}--{statement.span.end_line}."
                )
            raise

    def _handle_qubit_declaration(self, statement: ast.QubitDeclaration):
        name = statement.qubit.name
        if statement.size is not None:
            if not isinstance(statement.size, ast.IntegerLiteral):
                raise ValueError("Qubit declaration size must be an integer.")
            self.variables.add_array_variable(name, statement.size.value)
        else:
            self.variables.add_variable(name)

    def _handle_alias_statement(self, statement: ast.AliasStatement):
        if not isinstance(statement.target, ast.Identifier):
            raise ValueError("Alias target must be an identifier.")
        name = statement.target.name
        if isinstance(statement.value, ast.IndexExpression):
            collection, idx = get_collection_and_single_index(statement.value)
            if collection is None:
                raise ValueError("Array name must be an identifier.")
            if idx is None:
                raise ValueError("Alias index must be a single integer.")
            self.variables.add_alias(name, collection, idx)
        elif isinstance(statement.value, ast.Identifier):
            self.variables.add_alias(name, statement.value.name)
        else:
            raise ValueError("Alias value must be an identifier or index expression.")

    def _handle_quantum_gate(self, statement: ast.QuantumGate, root: Section):
        # todo: phase
        if statement.modifiers or statement.arguments or statement.duration:
            raise ValueError(
                "Gate modifiers, arguments, and duration not yet supported."
            )
        if not isinstance(statement.name, ast.Identifier):
            raise ValueError("Gate name must be an identifier.")
        name = statement.name.name
        qubit_indices = tuple(self._get_qubit_index(q) for q in statement.qubits)
        try:
            root.add(
                self.gate_store[self.gate_map.get(name, name), tuple(qubit_indices)]
            )
        except KeyError:
            raise ValueError(
                f"Gate '{name}' for qubit indices {qubit_indices} not found."
            )

    def _handle_include(self, statement: ast.Include):
        if statement.filename != "stdgates.inc":
            raise ValueError(
                f"Only 'stdgates.inc' is supported for include, found '{statement.filename}'."
            )

    def _handle_quantum_reset(self, statement: ast.QuantumReset, root: Section):
        # Although ``qubits`` is plural, only a single qubit is allowed.
        qubit_index = self._get_qubit_index(statement.qubits)
        try:
            root.add(
                self.gate_store[self.gate_map.get("reset", "reset"), (qubit_index,)]
            )
        except KeyError:
            raise ValueError(f"Reset gate for qubit index {qubit_index} not found.")

    def _get_qubit_index(self, q: Union[ast.IndexedIdentifier, ast.Identifier]):
        if isinstance(q, ast.Identifier):
            return self.variables.get_qubit_number(q.name)
        elif isinstance(q, ast.IndexedIdentifier):
            collection, idx = get_collection_and_single_index(q)
            if collection is None:
                raise ValueError("Qubit name must be an identifier.")
            if idx is None:
                raise ValueError("Qubit index must be a single integer.")
            return self.variables.get_qubit_number(collection, idx)
        else:
            raise ValueError("Qubit names must be identifiers or index expressions.")
