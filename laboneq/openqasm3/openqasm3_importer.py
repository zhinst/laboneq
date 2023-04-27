# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from typing import Optional, TextIO, Union

import openpulse
import openpulse.ast as ast

import openqasm3.visitor
from laboneq.dsl.experiment import Section
from laboneq.dsl.experiment.utils import id_generator
from laboneq.openqasm3.expression import eval_expression
from laboneq.openqasm3.gate_store import GateStore
from laboneq.openqasm3.openqasm_error import OpenQasmException
from laboneq.openqasm3.variable_store import VariableStore

ALLOWED_NODE_TYPES = {
    ast.Program,
    ast.QuantumGate,
    ast.Box,
    ast.Identifier,
    ast.IntegerLiteral,
    ast.FloatLiteral,
    ast.DurationLiteral,
    ast.BitstringLiteral,
    ast.BooleanLiteral,
    ast.ImaginaryLiteral,
    ast.BinaryExpression,
    ast.BinaryOperator,
    ast.UnaryExpression,
    ast.UnaryOperator,
    ast.Span,
    ast.QubitDeclaration,
    ast.AliasStatement,
    ast.IndexExpression,
    ast.Include,
    ast.RangeDefinition,
    ast.IndexedIdentifier,
    ast.QuantumReset,
}


class AllowedNodeTypesVisitor(openqasm3.visitor.QASMVisitor):
    def generic_visit(self, node: ast.QASMNode, context=None):
        if type(node) not in ALLOWED_NODE_TYPES:
            raise OpenQasmException(
                f"Node type {type(node)} not yet supported", mark=node.span
            )
        super().generic_visit(node, context)


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
        gate_store: GateStore,
    ):
        self.gate_store = gate_store
        self.scoped_variables = deque([VariableStore({})])

    def __call__(
        self,
        text: Optional[str] = None,
        file: Optional[TextIO] = None,
        filename: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> Section:
        if [arg is not None for arg in [text, file, filename, stream]].count(True) != 1:
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

    def _merge_scoped_variables(self):
        result = VariableStore()
        for vars in self.scoped_variables:
            result.variables.update(vars.variables)
            result.current_index += vars.current_index
        return result

    def _import_text(self, text) -> Section:
        tree = openpulse.parse(text)
        assert isinstance(tree, ast.Program)
        AllowedNodeTypesVisitor().visit(tree, None)
        try:
            return self.transpile(tree, uid_hint="root")
        except OpenQasmException as e:
            e.source = text
            raise

    def transpile(self, parent: Union[ast.Program, ast.Box], uid_hint="") -> Section:
        sect = Section(uid=id_generator(uid_hint))
        self.scoped_variables.append(VariableStore({}))

        try:
            body = parent.statements
        except AttributeError:
            body = parent.body

        for child in body:
            try:
                if isinstance(child, ast.QubitDeclaration):
                    self._handle_qubit_declaration(child)
                elif isinstance(child, ast.AliasStatement):
                    self._handle_alias_statement(child)
                elif isinstance(child, ast.Include):
                    self._handle_include(child)
                elif isinstance(child, ast.QuantumGate):
                    self._handle_quantum_gate(child, sect)
                elif isinstance(child, ast.Box):
                    self._handle_box(child, sect)
                elif isinstance(child, ast.QuantumReset):
                    self._handle_quantum_reset(child, sect)
                else:
                    raise OpenQasmException(
                        f"Statement type {type(child)} not supported",
                        mark=child.span,
                    )
            except OpenQasmException:
                raise
            except Exception as e:
                mark = child.span
                raise OpenQasmException("Failed to process statement", mark) from e
        return sect

    def _handle_qubit_declaration(self, statement: ast.QubitDeclaration):
        name = statement.qubit.name
        try:
            if statement.size is not None:
                if not isinstance(statement.size, ast.IntegerLiteral):
                    raise OpenQasmException(
                        "Qubit declaration size must be an integer.",
                        mark=statement.span,
                    )
                self.scoped_variables[-1].add_array_variable(name, statement.size.value)
            else:
                self.scoped_variables[-1].add_variable(name)
        except ValueError as e:
            raise OpenQasmException(str(e), mark=statement.span) from e

    def _handle_alias_statement(self, statement: ast.AliasStatement):
        if not isinstance(statement.target, ast.Identifier):
            raise OpenQasmException(
                "Alias target must be an identifier.", mark=statement.span
            )
        name = statement.target.name
        if isinstance(statement.value, ast.IndexExpression):
            collection, idx = get_collection_and_single_index(statement.value)
            if collection is None:
                raise OpenQasmException(
                    "Array name must be an identifier.", statement.span
                )
            if idx is None:
                raise OpenQasmException(
                    "Alias index must be a single integer.", mark=statement.span
                )
            try:
                self.scoped_variables[-1].add_alias(name, collection, idx)
            except ValueError as e:
                raise OpenQasmException(str(e), mark=statement.span) from e
        elif isinstance(statement.value, ast.Identifier):
            try:
                self.scoped_variables[-1].add_alias(name, statement.value.name)
            except ValueError as e:
                raise OpenQasmException(str(e), mark=statement.span) from e
        else:
            raise OpenQasmException(
                "Alias value must be an identifier or index expression.",
                mark=statement.span,
            )

    def _handle_quantum_gate(self, statement: ast.QuantumGate, parent: Section):
        args = tuple(eval_expression(arg) for arg in statement.arguments)
        if statement.modifiers or statement.duration:
            raise OpenQasmException(
                "Gate modifiers and duration not yet supported.",
                mark=statement.span,
            )
        if not isinstance(statement.name, ast.Identifier):
            raise OpenQasmException(
                "Gate name must be an identifier.", mark=statement.span
            )
        name = statement.name.name
        qubit_indices = tuple(self._get_qubit_index(q) for q in statement.qubits)
        try:
            parent.add(self.gate_store.lookup_gate(name, qubit_indices, args=args))
        except KeyError as e:
            raise OpenQasmException(
                f"Gate '{name}' for qubit indices {qubit_indices} not found.",
                mark=statement.span,
            ) from e

    def _handle_box(self, statement: ast.Box, parent: Section):
        if statement.duration:
            raise ValueError("Box duration not yet supported.")
        try:
            parent.add(self.transpile(statement, uid_hint="box"))
        except KeyError:
            raise ValueError(f"Unable to add box for {parent}.")

    def _handle_include(self, statement: ast.Include):
        if statement.filename != "stdgates.inc":
            raise OpenQasmException(
                f"Only 'stdgates.inc' is supported for include, found '{statement.filename}'.",
                mark=statement.span,
            )

    def _handle_quantum_reset(self, statement: ast.QuantumReset, parent: Section):
        # Although ``qubits`` is plural, only a single qubit is allowed.
        qubit_index = self._get_qubit_index(statement.qubits)
        try:
            parent.add(self.gate_store.lookup_gate("reset", (qubit_index,)))
        except KeyError as e:
            raise OpenQasmException(
                f"Reset gate for qubit index {qubit_index} not found.",
                mark=statement.span,
            ) from e

    def _get_qubit_index(self, q: Union[ast.IndexedIdentifier, ast.Identifier]):
        if isinstance(q, ast.Identifier):
            try:
                return self._merge_scoped_variables().get_qubit_number(q.name)
            except (KeyError, ValueError) as e:
                raise OpenQasmException(str(e), mark=q.span) from e
        elif isinstance(q, ast.IndexedIdentifier):
            collection, idx = get_collection_and_single_index(q)
            if collection is None:
                raise OpenQasmException(
                    "Qubit name must be an identifier.", mark=q.span
                )
            if idx is None:
                raise OpenQasmException(
                    "Qubit index must be a single integer.", mark=q.span
                )
            return self._merge_scoped_variables().get_qubit_number(collection, idx)
        else:
            raise OpenQasmException(
                "Qubit names must be identifiers or index expressions.", mark=q.span
            )
