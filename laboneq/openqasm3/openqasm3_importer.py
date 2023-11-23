# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa: I001

import math
import operator
from contextlib import contextmanager
from typing import Any, Optional, TextIO, Union

import openpulse
from openpulse import ast

import openqasm3.visitor
from laboneq._utils import id_generator
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl import LinearSweepParameter, SweepParameter
from laboneq.dsl.enums import AcquisitionType, AveragingMode, SectionAlignment
from laboneq.dsl.experiment import Experiment, Section, Sweep
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
    ast.CalibrationGrammarDeclaration,
    ast.CalibrationStatement,
    ast.ClassicalDeclaration,
    ast.ConstantDeclaration,
    ast.Concatenation,
    ast.DiscreteSet,
    ast.ExpressionStatement,
    ast.ExternDeclaration,
    ast.FrameType,
    ast.FunctionCall,
    ast.Identifier,
    ast.Include,
    ast.Program,
    ast.RangeDefinition,
    ast.Span,
    ast.ClassicalAssignment,
    ast.ForInLoop,
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
    ast.UintType,
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
        qubits: dict[str, Qubit, Transmon] | None = None,
    ):
        self.gate_store = gate_store
        self.dsl_qubits = qubits
        self.scope = NamespaceNest()

    def __call__(
        self,
        text: str | None = None,
        file: TextIO | None = None,
        filename: str | None = None,
        stream: TextIO | None = None,
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

    def _parse_valid_tree(self, text) -> ast.Program:
        tree = openpulse.parse(text)
        assert isinstance(tree, ast.Program)
        return tree

    def _workaround_extern_port(self, text: str) -> str:
        new_text = ""
        for line in text.splitlines():
            if line.strip().startswith("extern port "):
                name = line.strip().split(" ")[2]
                self.scope.current.declare_classical_value(name, 0)
            else:
                new_text += line + "\n"
        return new_text

    def _import_text(self, text) -> Section:
        text = self._workaround_extern_port(text)
        tree = self._parse_valid_tree(text)
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

    def transpile(
        self, parent: Union[ast.Program, ast.Box, ast.ForInLoop], uid_hint=""
    ) -> Section:
        sect = Section(uid=id_generator(uid_hint))

        if isinstance(parent, ast.Program):
            body = parent.statements
        elif isinstance(parent, (ast.Box, ast.CalibrationStatement)):
            body = parent.body
        elif isinstance(parent, ast.ForInLoop):
            body = parent.block
        else:
            msg = f"Unsupported block type {type(parent)!r}"
            raise OpenQasmException(msg, mark=parent.span)

        for child in body:
            subsect = None
            try:
                if isinstance(child, ast.QubitDeclaration):
                    self._handle_qubit_declaration(child)
                elif isinstance(child, ast.ClassicalDeclaration):
                    self._handle_classical_declaration(child)
                elif isinstance(child, ast.ConstantDeclaration):
                    self._handle_constant_declaration(child)
                elif isinstance(child, ast.ExternDeclaration):
                    self._handle_extern_declaration(child)
                elif isinstance(child, ast.AliasStatement):
                    self._handle_alias_statement(child)
                elif isinstance(child, ast.Include):
                    self._handle_include(child)
                elif isinstance(child, ast.CalibrationGrammarDeclaration):
                    self._handle_calibration_grammar(child)
                elif isinstance(child, ast.CalibrationStatement):
                    subsect = self._handle_calibration(child)
                elif isinstance(child, ast.ExpressionStatement):
                    subsect = self._handle_cal_expression(child)
                elif isinstance(child, ast.QuantumGate):
                    subsect = self._handle_quantum_gate(child)
                elif isinstance(child, ast.Box):
                    subsect = self._handle_box(child)
                elif isinstance(child, ast.QuantumBarrier):
                    subsect = self._handle_barrier(child)
                elif isinstance(child, ast.DelayInstruction):
                    subsect = self._handle_delay_instruction(child)
                elif isinstance(child, ast.ForInLoop):
                    subsect = self._handle_for_in_loop(child)
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

    def _handle_qubit_declaration(self, statement: ast.QubitDeclaration) -> None:
        name = statement.qubit.name
        try:
            if statement.size is not None:
                try:
                    size = eval_expression(
                        statement.size, namespace=self.scope, type=int
                    )
                except Exception:
                    msg = "Qubit declaration size must evaluate to an integer."
                    raise OpenQasmException(msg, mark=statement.span) from None

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

    def _handle_classical_declaration(
        self, statement: ast.ClassicalDeclaration
    ) -> None:
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
        elif isinstance(statement.type, ast.FrameType):
            init = statement.init_expression
            if not isinstance(init, ast.FunctionCall) or init.name.name != "newframe":
                msg = "Frame type initializer must be a 'newframe' function call."
                raise OpenQasmException(msg, mark=statement.span)
            name = statement.identifier.name
            port = statement.init_expression.arguments[0].name
            freq = statement.init_expression.arguments[1].value
            phase = statement.init_expression.arguments[2].value
            self.scope.current.declare_frame(name, port, freq, phase)
        else:
            value = eval_expression(statement.init_expression, namespace=self.scope)
            self.scope.current.declare_classical_value(name, value)

    def _handle_constant_declaration(self, statement: ast.ConstantDeclaration) -> None:
        name = statement.identifier.name
        value = eval_expression(statement.init_expression, namespace=self.scope)
        self.scope.current.declare_classical_value(name, value)

    # TODO: currently unused due to 'extern port' workaround
    def _handle_extern_declaration(self, statement: ast.ExternDeclaration) -> None:
        name = statement.identifier.name
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
        args = tuple(
            eval_expression(arg, namespace=self.scope) for arg in statement.arguments
        )
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

    def _handle_calibration_grammar(
        self, statement: ast.CalibrationGrammarDeclaration
    ) -> None:
        if statement.name != "openpulse":
            msg = f"Only 'openpulse' is supported for defcalgrammar, found '{statement.name}'."
            raise OpenQasmException(msg, mark=statement.span)

    def _handle_calibration(self, statement: ast.CalibrationStatement):
        return self.transpile(statement, uid_hint="calibration")

    def _handle_cal_expression(self, statement: ast.ExpressionStatement):
        expr = statement.expression
        if isinstance(expr, ast.FunctionCall) and expr.name.name == "play":
            return self._handle_play(expr)
        else:
            msg = "Currently only 'play' is supported as a calibration expression."
            raise OpenQasmException(msg, mark=statement.span)

    def _handle_play(self, expr: ast.FunctionCall):
        frame = eval_expression(expr.arguments[0], namespace=self.scope)
        arg1 = expr.arguments[1]
        if isinstance(arg1, ast.FunctionCall):
            if arg1.name.name == "scale":
                waveform = arg1.arguments[1].name
                amplitude = eval_expression(arg1.arguments[0], namespace=self.scope)
            else:
                msg = "Currently only 'scale' is supported as a play waveform modifier function."
                raise OpenQasmException(msg, mark=expr.span)
        else:
            waveform = arg1.name
            amplitude = None

        pulse = self.gate_store.lookup_waveform(waveform)
        drive_line = self.gate_store.ports[frame.port]

        sect = Section(uid=id_generator(f"play_{frame.port}"))
        sect.play(
            signal=drive_line,
            pulse=pulse,
            amplitude=amplitude,
        )
        return sect

    def _handle_delay_instruction(self, statement: ast.DelayInstruction):
        qubits = statement.qubits
        duration = eval_expression(
            statement.duration, namespace=self.scope, type=(float, SweepParameter)
        )
        qubit_names = [
            eval_expression(qubit, namespace=self.scope).canonical_name
            for qubit in qubits
        ]
        qubits_str = "_".join(qubit_names)
        delay_section = Section(uid=id_generator(f"{qubits_str}_delay"))
        for qubit in qubit_names:
            dsl_qubit = self.dsl_qubits[qubit]
            # TODO: I think we might be able to remove this loop with
            #       the new qubit class.
            # TODO: Is it correct to delay on only one line?
            # TODO: What should happen for custom qubit types?
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

    def _handle_for_in_loop(self, statement: ast.ForInLoop):
        loop_var = statement.identifier.name
        loop_set_decl = statement.set_declaration

        if isinstance(loop_set_decl, ast.RangeDefinition):
            start = eval_expression(loop_set_decl.start, namespace=self.scope)
            stop = eval_expression(loop_set_decl.end, namespace=self.scope)
            step = eval_expression(loop_set_decl.step, namespace=self.scope) or 1
            count = math.floor(((stop - start) / step) + 1)
            sweep_param = LinearSweepParameter(
                uid=id_generator("sweep_parameter"),
                start=start,
                stop=stop,
                count=count,
            )
        else:
            raise OpenQasmException(
                f"Loop set declaration type {type(loop_set_decl)!r} is not"
                f" yet supported by the LabOne Q OpenQASM importer.",
                mark=statement.set_declaration.span,
            )

        sweep = Sweep(
            uid=id_generator("sweep"),
            parameters=[sweep_param],
            alignment=SectionAlignment.LEFT,
        )

        with self._new_scope():
            self.scope.current.declare_classical_value(loop_var, sweep_param)
            subsect = self.transpile(statement, uid_hint="block")
            sweep.add(subsect)

        return sweep

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


def exp_from_qasm(
    program: str,
    qubits: dict[str, Qubit],
    gate_store: GateStore,
    count: int = 1,
    averaging_mode: AveragingMode = AveragingMode.CYCLIC,
    acquisition_type: AcquisitionType = AcquisitionType.INTEGRATION,
):
    """Create an experiment from an OpenQASM program.

    Args:
        program:
            OpenQASM program
        qubits:
            Map from OpenQASM qubit names to LabOne Q DSL Qubit objects
        gate_store:
            Map from OpenQASM gate names to LabOne Q DSL Gate objects
        count:
            The number of acquire iterations.
        averaging_mode:
            The mode of how to average the acquired data.
        acquisition_type:
            The type of acquisition to perform.
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
    with exp.acquire_loop_rt(
        count=count,
        averaging_mode=averaging_mode,
        acquisition_type=acquisition_type,
    ) as loop:
        loop.add(qasm_section)

    return exp
