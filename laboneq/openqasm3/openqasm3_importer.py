# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator
from collections import deque
from contextlib import contextmanager
from typing import Optional, TextIO, Union

import openpulse
import openpulse.ast as ast

import openqasm3.visitor
from laboneq.dsl.experiment import Section
from laboneq.dsl.experiment.utils import id_generator
from laboneq.openqasm3.expression import eval_expression, eval_lvalue
from laboneq.openqasm3.gate_store import GateStore
from laboneq.openqasm3.namespace import Namespace, QubitRef
from laboneq.openqasm3.openqasm_error import OpenQasmException
from laboneq.openqasm3.signal_store import SignalStore

ALLOWED_NODE_TYPES = {
    # quantum logic
    ast.Box,
    ast.DelayInstruction,
    ast.QuantumBarrier,
    ast.QuantumGate,
    ast.QuantumReset,
    ast.QubitDeclaration,
    # auxiliary
    ast.AliasStatement,
    ast.ClassicalDeclaration,
    ast.Concatenation,
    ast.DiscreteSet,
    ast.Identifier,
    ast.Include,
    ast.Program,
    ast.RangeDefinition,
    ast.Span,
    ast.ClassicalAssignment,
    # expressions
    ast.BinaryExpression,
    ast.BinaryOperator,
    ast.IndexedIdentifier,
    ast.IndexExpression,
    ast.UnaryExpression,
    ast.UnaryOperator,
    ast.AssignmentOperator,
    # literals
    ast.BitstringLiteral,
    ast.BooleanLiteral,
    ast.DurationLiteral,
    ast.FloatLiteral,
    ast.ImaginaryLiteral,
    ast.IntegerLiteral,
    # types
    ast.IntType,
    ast.FloatType,
    ast.BoolType,
    ast.DurationType,
    ast.BitType,
}


class AllowedNodeTypesVisitor(openqasm3.visitor.QASMVisitor):
    def generic_visit(self, node: ast.QASMNode, context=None):
        if type(node) not in ALLOWED_NODE_TYPES:
            raise OpenQasmException(
                f"Node type {type(node)} not yet supported", mark=node.span
            )
        super().generic_visit(node, context)


class OpenQasm3Importer:
    def __init__(
        self, gate_store: GateStore, signal_store: Optional[SignalStore] = None
    ):
        self.gate_store = gate_store
        self.signal_store = (
            signal_store if signal_store is not None else SignalStore({})
        )
        self.scoped_variables = deque([Namespace()])

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

    def _import_text(self, text) -> Section:
        tree = openpulse.parse(text)
        assert isinstance(tree, ast.Program)
        AllowedNodeTypesVisitor().visit(tree, None)
        try:
            root = self.transpile(tree, uid_hint="root")
        except OpenQasmException as e:
            e.source = text
            raise
        self.signal_store.leftover_raise()
        return root

    @contextmanager
    def _new_scope(self):
        parent_namespace = self.scoped_variables[-1]
        self.scoped_variables.append(Namespace(parent=parent_namespace))
        yield
        self.scoped_variables.pop()

    def transpile(self, parent: Union[ast.Program, ast.Box], uid_hint="") -> Section:
        sect = Section(uid=id_generator(uid_hint))

        try:
            body = parent.statements
        except AttributeError:
            body = parent.body

        for child in body:
            subsect = None
            try:
                if isinstance(child, ast.QubitDeclaration):
                    self._handle_qubit_declaration(child)
                elif isinstance(child, ast.ClassicalDeclaration):
                    self._handle_classical_declaration(child)
                elif isinstance(child, ast.AliasStatement):
                    self._handle_alias_statement(child)
                elif isinstance(child, ast.Include):
                    self._handle_include(child)
                elif isinstance(child, ast.QuantumGate):
                    subsect = self._handle_quantum_gate(child)
                elif isinstance(child, ast.Box):
                    subsect = self._handle_box(child)
                elif isinstance(child, ast.QuantumBarrier):
                    subsect = self._handle_barrier(child)
                elif isinstance(child, ast.QuantumReset):
                    subsect = self._handle_quantum_reset(child)
                elif isinstance(child, ast.DelayInstruction):
                    subsect = self._handle_delay_instruction(child)
                elif isinstance(child, ast.ClassicalAssignment):
                    self._handle_assignment(child)
                else:
                    raise OpenQasmException(
                        f"Statement type {type(child)} not supported",
                        mark=child.span,
                    )
                if subsect is not None:
                    sect.add(subsect)
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
                try:
                    size = eval_expression(
                        statement.size, namespace=self.scoped_variables[-1], type=int
                    )
                except Exception:
                    raise OpenQasmException(
                        "Qubit declaration size must evaluate to an integer.",
                        mark=statement.span,
                    )

                # declare the individual qubits...
                qubits = [
                    self.scoped_variables[-1].declare_qubit(f"{name}[{i}]")
                    for i in range(size)
                ]
                # ... as well as a list aliasing them
                self.scoped_variables[-1].declare_reference(name, qubits)
            else:
                self.scoped_variables[-1].declare_qubit(name)
        except ValueError as e:
            raise OpenQasmException(str(e), mark=statement.span) from e
        except OpenQasmException as e:
            e.mark = statement.span
            raise

    def _handle_classical_declaration(self, statement: ast.ClassicalDeclaration):
        name = statement.identifier.name
        if isinstance(statement.type, ast.BitType):
            value = eval_expression(
                statement.init_expression, namespace=self.scoped_variables[-1], type=int
            )
            size = statement.type.size
            if size is not None:
                size = eval_expression(
                    size, namespace=self.scoped_variables[-1], type=int
                )

                # declare the individual bits...
                bits = [
                    self.scoped_variables[-1].declare_classical_value(
                        f"{name}[{i}]", value=bool((value >> i) & 1)
                    )
                    for i in range(size)
                ]
                # ... as well as a list aliasing them
                self.scoped_variables[-1].declare_reference(name, bits)
            else:
                self.scoped_variables[-1].declare_classical_value(name, value)
        else:
            value = eval_expression(
                statement.init_expression, namespace=self.scoped_variables[-1]
            )
            self.scoped_variables[-1].declare_classical_value(name, value)

    def _handle_alias_statement(self, statement: ast.AliasStatement):
        if not isinstance(statement.target, ast.Identifier):
            raise OpenQasmException(
                "Alias target must be an identifier.", mark=statement.span
            )
        name = statement.target.name

        try:
            value = eval_lvalue(statement.value, namespace=self.scoped_variables[-1])
        except OpenQasmException:
            raise
        except Exception as e:
            raise OpenQasmException(
                "Invalid alias value", mark=statement.value.span
            ) from e
        try:
            self.scoped_variables[-1].declare_reference(name, value)
        except OpenQasmException as e:
            e.mark = statement.span
            raise

    def _handle_quantum_gate(self, statement: ast.QuantumGate):
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
        qubit_names = []
        for q in statement.qubits:
            qubit = eval_expression(q, namespace=self.scoped_variables[-1])
            try:
                qubit_names.append(qubit.canonical_name)
            except AttributeError as e:
                raise OpenQasmException(
                    f"Qubit expected, got '{type(qubit).__name__}'", mark=q.span
                ) from e
        qubit_names = tuple(qubit_names)
        try:
            return self.gate_store.lookup_gate(name, qubit_names, args=args)
        except KeyError as e:
            raise OpenQasmException(
                f"Gate '{name}' for qubit(s) {qubit_names} not found.",
                mark=statement.span,
            ) from e

    def _handle_box(self, statement: ast.Box):
        if statement.duration:
            raise ValueError("Box duration not yet supported.")
        with self._new_scope():
            return self.transpile(statement, uid_hint="box")

    def _handle_barrier(self, statement: ast.QuantumBarrier):
        sect = Section(uid=id_generator("barrier"), length=0)
        reserved_qubits = [
            eval_expression(qubit, namespace=self.scoped_variables[-1]).canonical_name
            for qubit in statement.qubits
        ]
        reserved_signals = set()
        if not reserved_qubits:  # get all signals
            for each_qubit_signals in self.signal_store.user_map.values():
                for signal in each_qubit_signals:
                    reserved_signals.add(signal.exp_signal)
        for qubit in reserved_qubits:  # get only selected signals
            for signal in self.signal_store.user_map[qubit]:
                reserved_signals.add(signal.exp_signal)
        for exp_signal in reserved_signals:
            sect.reserve(exp_signal)
        return sect

    def _handle_include(self, statement: ast.Include):
        if statement.filename != "stdgates.inc":
            raise OpenQasmException(
                f"Only 'stdgates.inc' is supported for include, found '{statement.filename}'.",
                mark=statement.span,
            )

    def _handle_quantum_reset(self, statement: ast.QuantumReset):
        # Although ``qubits`` is plural, only a single qubit is allowed.
        qubit_name = eval_expression(
            statement.qubits, namespace=self.scoped_variables[-1]
        ).canonical_name
        try:
            return self.gate_store.lookup_gate("reset", (qubit_name,))
        except KeyError as e:
            raise OpenQasmException(
                f"Reset gate for qubit '{qubit_name}' not found.",
                mark=statement.span,
            ) from e

    def _handle_delay_instruction(self, statement: ast.DelayInstruction):
        qubits = statement.qubits
        duration = eval_expression(
            statement.duration, namespace=self.scoped_variables[-1], type=float
        )
        qubit_names = [
            eval_expression(qubit, namespace=self.scoped_variables[-1]).canonical_name
            for qubit in qubits
        ]
        qubits_str = "_".join(qubit_names)
        delay_section = Section(
            uid=id_generator(f"{qubits_str}_delay_{duration * 1e9:.0f}ns")
        )
        for qubit in qubit_names:
            # todo: TBD only delaying drive signal?
            delay_section.delay(signal=f"{qubit}_drive", time=duration)
        return delay_section

    def _handle_assignment(self, statement: ast.ClassicalAssignment):
        lvalue = eval_lvalue(statement.lvalue, self.scoped_variables[-1])
        if isinstance(lvalue, QubitRef):
            raise OpenQasmException(f"Cannot assign to qubit '{lvalue.canonical_name}'")
        if isinstance(lvalue, list):
            raise OpenQasmException("Cannot assign to arrays")
        ops = {
            "=": lambda a, b: b,
            "*=": operator.mul,
            "/=": operator.truediv,
            "+=": operator.add,
            "-=": operator.sub,
        }
        try:
            op = ops[statement.op.name]
        except KeyError as e:
            raise OpenQasmException(
                "Unsupported assignment operator", mark=statement.span
            ) from e
        rvalue = eval_expression(statement.rvalue, namespace=self.scoped_variables[-1])
        lvalue.value = op(lvalue.value, rvalue)
