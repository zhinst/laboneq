# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa: I001

import operator
from contextlib import contextmanager
from typing import Any, Optional, TextIO, Union

import openpulse
from openpulse import ast

import openqasm3.visitor
from laboneq._utils import id_generator
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.experiment import Experiment, Section
from laboneq.dsl.quantum.quantum_element import SignalType
from laboneq.dsl.quantum.qubit import Qubit
from laboneq.dsl.quantum.transmon import Transmon
from laboneq.openqasm3.expression import eval_expression, eval_lvalue
from laboneq.openqasm3.gate_store import GateStore
from laboneq.openqasm3.namespace import ClassicalRef, NamespaceNest, QubitRef
from laboneq.openqasm3.openqasm_error import OpenQasmException

ALLOWED_NODE_TYPES = {
    # quantum logic
    ast.Box,
    ast.DelayInstruction,
    ast.QuantumBarrier,
    ast.QuantumGate,
    ast.QuantumReset,
    ast.QubitDeclaration,
    ast.QuantumMeasurementStatement,
    ast.QuantumMeasurement,
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


class MeasurementResult:
    pass


class _AllowedNodeTypesVisitor(openqasm3.visitor.QASMVisitor):
    def generic_visit(self, node: ast.QASMNode, context: Optional[Any] = None) -> None:
        if type(node) not in ALLOWED_NODE_TYPES:
            msg = f"Node type {type(node)} not yet supported"
            raise OpenQasmException(msg, mark=node.span)
        super().generic_visit(node, context)


class OpenQasm3Importer:
    def __init__(
        self,
        gate_store: GateStore,
        qubits: dict[str, Qubit, Transmon] = None,
    ):
        self.gate_store = gate_store
        self.dsl_qubits = qubits
        self.scope = NamespaceNest()

    def __call__(
        self,
        text: Optional[str] = None,
        file: Optional[TextIO] = None,
        filename: Optional[str] = None,
        stream: Optional[TextIO] = None,
    ) -> Section:
        if [arg is not None for arg in [text, file, filename, stream]].count(True) != 1:
            msg = "Must specify exactly one of text, file, filename, or stream"
            raise ValueError(msg)
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
        _AllowedNodeTypesVisitor().visit(tree, None)
        try:
            root = self.transpile(tree, uid_hint="root")
        except OpenQasmException as e:
            e.source = text
            raise
        return root

    @contextmanager
    def _new_scope(self):
        self.scope.open()
        yield
        self.scope.close()

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
                elif isinstance(child, ast.DelayInstruction):
                    subsect = self._handle_delay_instruction(child)
                elif isinstance(child, ast.ClassicalAssignment):
                    self._handle_assignment(child)
                elif isinstance(child, ast.QuantumMeasurementStatement):
                    subsect = self._handle_measurement(child)
                elif isinstance(child, ast.QuantumReset):
                    subsect = self._handle_quantum_reset(child)
                else:
                    msg = f"Statement type {type(child)} not supported"
                    raise OpenQasmException(msg, mark=child.span)
            except OpenQasmException:
                raise
            except Exception as e:
                msg = "Failed to process statement"
                mark = child.span
                raise OpenQasmException(msg, mark) from e
            if subsect is not None:
                sect.add(subsect)

        return sect

    def _handle_qubit_declaration(self, statement: ast.QubitDeclaration):
        name = statement.qubit.name
        try:
            if statement.size is not None:
                try:
                    size = eval_expression(
                        statement.size, namespace=self.scope, type=int
                    )
                except Exception:
                    msg = "Qubit declaration size must evaluate to an integer."
                    raise OpenQasmException(msg, mark=statement.span)

                # declare the individual qubits...
                qubits = [
                    self.scope.current.declare_qubit(f"{name}[{i}]")
                    for i in range(size)
                ]
                # ... as well as a list aliasing them
                self.scope.current.declare_reference(name, qubits)
            else:
                self.scope.current.declare_qubit(name)
        except ValueError as e:
            raise OpenQasmException(str(e), mark=statement.span) from e
        except OpenQasmException as e:
            e.mark = statement.span
            raise

    def _handle_classical_declaration(self, statement: ast.ClassicalDeclaration):
        name = statement.identifier.name
        if isinstance(statement.type, ast.BitType):
            if statement.init_expression is not None:
                value = eval_expression(
                    statement.init_expression,
                    namespace=self.scope,
                    type=int,
                )
            else:
                value = None
            size = statement.type.size
            if size is not None:
                size = eval_expression(size, namespace=self.scope, type=int)

                # declare the individual bits...
                bits = [
                    self.scope.current.declare_classical_value(
                        f"{name}[{i}]",
                        value=bool((value >> i) & 1) if value is not None else None,
                    )
                    for i in range(size)
                ]
                # ... as well as a list aliasing them
                self.scope.current.declare_reference(name, bits)
            else:
                self.scope.current.declare_classical_value(name, value)
        else:
            value = eval_expression(statement.init_expression, namespace=self.scope)
            self.scope.current.declare_classical_value(name, value)

    def _handle_alias_statement(self, statement: ast.AliasStatement):
        if not isinstance(statement.target, ast.Identifier):
            msg = "Alias target must be an identifier."
            raise OpenQasmException(msg, mark=statement.span)
        name = statement.target.name

        try:
            value = eval_lvalue(statement.value, namespace=self.scope)
        except OpenQasmException:
            raise
        except Exception as e:
            msg = "Invalid alias value"
            raise OpenQasmException(msg, mark=statement.value.span) from e
        try:
            self.scope.current.declare_reference(name, value)
        except OpenQasmException as e:
            e.mark = statement.span
            raise

    def _handle_quantum_gate(self, statement: ast.QuantumGate):
        args = tuple(eval_expression(arg) for arg in statement.arguments)
        if statement.modifiers or statement.duration:
            msg = "Gate modifiers and duration not yet supported."
            raise OpenQasmException(msg, mark=statement.span)
        if not isinstance(statement.name, ast.Identifier):
            msg = "Gate name must be an identifier."
            raise OpenQasmException(msg, mark=statement.span)
        name = statement.name.name
        qubit_names = []
        for q in statement.qubits:
            qubit = eval_expression(q, namespace=self.scope)
            try:
                qubit_names.append(qubit.canonical_name)
            except AttributeError as e:
                msg = f"Qubit expected, got '{type(qubit).__name__}'"
                raise OpenQasmException(msg, mark=q.span) from e
        qubit_names = tuple(qubit_names)
        try:
            return self.gate_store.lookup_gate(name, qubit_names, args=args)
        except KeyError as e:
            gates = ", ".join(
                f"{gate[0]} for {gate[1]}" for gate in self.gate_store.gates
            )
            msg = f"Gate '{name}' for qubit(s) {qubit_names} not found.\nAvailable gates: {gates}"
            raise OpenQasmException(msg, mark=statement.span) from e

    def _handle_box(self, statement: ast.Box):
        if statement.duration:
            raise ValueError("Box duration not yet supported.")
        with self._new_scope():
            return self.transpile(statement, uid_hint="box")

    def _handle_barrier(self, statement: ast.QuantumBarrier):
        sect = Section(uid=id_generator("barrier"), length=0)

        reserved_qubits = [
            self.dsl_qubits[eval_expression(qubit, namespace=self.scope).canonical_name]
            for qubit in statement.qubits
        ]
        if not reserved_qubits:
            reserved_qubits = self.dsl_qubits.values()  # reserve all qubits

        reserved_signals = set()
        for qubit in reserved_qubits:
            for exp_signal in qubit.experiment_signals():
                reserved_signals.add(exp_signal.mapped_logical_signal_path)
        for signal in reserved_signals:
            sect.reserve(signal)

        return sect

    def _handle_include(self, statement: ast.Include) -> None:
        if statement.filename != "stdgates.inc":
            msg = f"Only 'stdgates.inc' is supported for include, found '{statement.filename}'."
            raise OpenQasmException(msg, mark=statement.span)

    def _handle_delay_instruction(self, statement: ast.DelayInstruction):
        qubits = statement.qubits
        duration = eval_expression(statement.duration, namespace=self.scope, type=float)
        qubit_names = [
            eval_expression(qubit, namespace=self.scope).canonical_name
            for qubit in qubits
        ]
        qubits_str = "_".join(qubit_names)
        delay_section = Section(
            uid=id_generator(f"{qubits_str}_delay_{duration * 1e9:.0f}ns")
        )
        for qubit in qubit_names:
            dsl_qubit = self.dsl_qubits[qubit]
            for role, sig in dsl_qubit.experiment_signals(with_types=True):
                if role != SignalType.DRIVE:
                    continue
                delay_section.delay(sig.mapped_logical_signal_path, time=duration)
        if not delay_section.children:
            msg = (
                f"Unable to apply delay to {qubit_names} due to missing drive signals."
            )
            raise OpenQasmException(msg, mark=statement.span)

        return delay_section

    def _handle_assignment(self, statement: ast.ClassicalAssignment):
        lvalue = eval_lvalue(statement.lvalue, namespace=self.scope)
        if isinstance(lvalue, QubitRef):
            msg = f"Cannot assign to qubit '{lvalue.canonical_name}'"
            raise OpenQasmException(msg)
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
            msg = "Unsupported assignment operator"
            raise OpenQasmException(msg, mark=statement.span) from e
        rvalue = eval_expression(statement.rvalue, namespace=self.scope)
        lvalue.value = op(lvalue.value, rvalue)

    def _handle_measurement(self, statement: ast.QuantumMeasurementStatement):
        qubits = eval_expression(statement.measure.qubit, namespace=self.scope)
        bits = statement.target
        if bits is None:
            raise OpenQasmException(
                "Measurement must be assigned to a classical bit", mark=statement.span
            )
        bits = eval_lvalue(statement.target, namespace=self.scope)
        if isinstance(qubits, list):
            err_msg = None
            if not isinstance(bits, list):
                err_msg = "Both bits and qubits must be either scalar or registers."
            if not len(bits) == len(qubits):
                err_msg = "Bit and qubit registers must be same length"
            if err_msg is not None:
                raise OpenQasmException(err_msg, statement.span)
        else:
            bits = [bits]
            qubits = [qubits]

        assert all(isinstance(q, QubitRef) for q in qubits)
        assert all(isinstance(b, ClassicalRef) for b in bits)

        # Build the section
        s = Section(uid=id_generator("measurement"))
        for q, b in zip(qubits, bits):
            handle_name = b.canonical_name
            qubit_name = q.canonical_name
            try:
                gate_section = self.gate_store.lookup_gate(
                    "measure", (qubit_name,), kwargs={"handle": handle_name}
                )
            except KeyError as e:
                raise OpenQasmException(
                    f"No measurement operation defined for qubit '{qubit_name}'",
                    mark=statement.span,
                ) from e
            s.add(gate_section)

            # Set the bit to a special value to disallow compile time arithmetic
            b.value = MeasurementResult()
        return s

    def _handle_quantum_reset(self, statement: ast.QuantumReset):
        # Although ``qubits`` is plural, only a single qubit is allowed.
        qubit_name = eval_expression(
            statement.qubits, namespace=self.scope
        ).canonical_name
        try:
            return self.gate_store.lookup_gate("reset", (qubit_name,))
        except KeyError as e:
            msg = f"Reset gate for qubit '{qubit_name}' not found."
            raise OpenQasmException(msg, mark=statement.span) from e


def exp_from_qasm(program: str, qubits: dict[str, Qubit], gate_store: GateStore):
    """Create an experiment from an OpenQASM program.

    Args:
        program:
            OpenQASM program
        qubits:
            Map from OpenQASM qubit names to LabOne Q DSL Qubit objects
        gate_store:
            Map from OpenQASM gate names to LabOne Q DSL Gate objects
    """
    importer = OpenQasm3Importer(qubits=qubits, gate_store=gate_store)
    qasm_section = importer(text=program)

    signals = []
    for qubit in qubits.values():
        for exp_signal in qubit.experiment_signals():
            if exp_signal in signals:
                msg = f"Signal with id {exp_signal.uid} already assigned."
                raise LabOneQException(msg)
            signals.append(exp_signal)

    # TODO: feed qubits directly to experiment when feature is implemented
    exp = Experiment(signals=signals)
    with exp.acquire_loop_rt(count=1) as loop:
        loop.add(qasm_section)

    return exp
