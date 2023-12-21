# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations  # noqa: I001

import math
import operator
from contextlib import contextmanager
from typing import Any, Callable, Optional, TextIO, Union

import openpulse
import openqasm3.visitor
from openpulse import ast

from laboneq._utils import id_generator
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl import LinearSweepParameter, SweepParameter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.enums import (
    AcquisitionType,
    AveragingMode,
    ModulationType,
    SectionAlignment,
)
from laboneq.dsl.experiment import Experiment, Section, Sweep
from laboneq.dsl.quantum.quantum_element import SignalType
from laboneq.dsl.quantum.qubit import Qubit
from laboneq.dsl.quantum.transmon import Transmon
from laboneq.openqasm3.expression import eval_expression, eval_lvalue
from laboneq.openqasm3.gate_store import GateStore
from laboneq.openqasm3.namespace import ClassicalRef, Frame, NamespaceNest, QubitRef
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
    ast.CalibrationDefinition,
    ast.ConstantDeclaration,
    ast.Concatenation,
    ast.DiscreteSet,
    ast.ExpressionStatement,
    ast.ExternDeclaration,
    ast.ExternArgument,
    ast.FrameType,
    ast.FunctionCall,
    ast.Identifier,
    ast.Include,
    ast.IODeclaration,
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
    ast.BitType,
    ast.BoolType,
    ast.ComplexType,
    ast.DurationType,
    ast.FloatType,
    ast.IntType,
    ast.UintType,
    # openpulse
    openpulse.ast.WaveformType,
}


class MeasurementResult:
    """An internal holder for measurement results."""

    def __init__(self, handle: str):
        self.handle = handle


class ExternResult:
    """A class for holding the result of an extern function that
    needs to play a section, perform a measurement or other operations in
    addition to returning a result.

    Arguments:
        result:
            The result returned by the extern function.
        handle:
            The measurement handle that holds the result returned by
            the extern function. Only one of handle or result
            may be specified.
        section:
            The section the extern function would like to have played at the
            point in the OpenQASM function where it is called. Sections cannot
            be played inside an expression -- the call to the extern must be
            directly a statement or the right-hand side of an assignment.

    Attributes:
        result (Any):
            The result returned by the extern funciton.
        handle (str | None):
            The measurement handle holding the result.
        section (Section | None):
            The section the extern function wishes to play.
    """

    def __init__(self, result=None, handle=None, section=None):
        self.result = result
        self.handle = handle
        self.section = section
        if self.result is not None and self.handle is not None:
            raise OpenQasmException(
                f"An external function may return either a result or a handle"
                f", not both: {self!r}"
            )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}"
            f" result={self.result!r}"
            f" handle={self.handle!r}"
            f" has_section={self.section is not None}>"
        )


class _AllowedNodeTypesVisitor(openqasm3.visitor.QASMVisitor):
    def generic_visit(self, node: ast.QASMNode, context: Optional[Any] = None) -> None:
        if type(node) not in ALLOWED_NODE_TYPES:
            msg = f"Node type {type(node)} not yet supported"
            raise OpenQasmException(msg, mark=node.span)
        super().generic_visit(node, context)


def _convert_openpulse_span(base_span, relative_span, prefix_length):
    """
    Add a relative OpenPulse span to a base OpenQASM parser
    span.

    When parsing `cal` and `defcal` blocks the spans returned
    by the OpenPulse parser are relative to the block, rather
    than to the start of the OpenQASM program. This functions
    adds the relative span to the base on.

    The prefix length specifies how many characters from the
    start of the base_span (i.e. start to the relevant statement)
    to the start of the OpenPulse block (i.e. character after the
    opening curly bracket).
    """
    # TODO: File a bug report on the openpulse library:
    #
    # The OpenPulse parser throws away the details of
    # how 'cal {' was written, so we have to guess that
    # it was written exactly as above.
    first_column_fudge = prefix_length + 1
    start_column = relative_span.start_column
    if relative_span.start_line == 1:
        start_column += base_span.start_column + first_column_fudge - 1
    end_column = relative_span.end_column
    if relative_span.end_line == 1:
        end_column += base_span.start_column + first_column_fudge - 1
    return ast.Span(
        start_line=base_span.start_line + relative_span.start_line - 1,
        start_column=start_column,
        end_line=base_span.start_line + relative_span.end_line - 1,
        end_column=end_column,
    )


class OpenQasm3Importer:
    def __init__(
        self,
        gate_store: GateStore,
        qubits: dict[str, Union[Qubit, Transmon]] | None = None,
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable] | None = None,
    ):
        # TODO: The gate store and qubits may be considered
        #       part of the target hardware platform, but
        #       the scope is definitely for a specific program
        #       being compiled and the inputs are for a specific
        #       execution of a program. We need to re-structure
        #       the importer to allow the scope and inputs
        #       to be specified at appropriate points (regardless
        #       of what the L1Q DSL is currently capable of).
        self.gate_store = gate_store
        self.dsl_qubits = qubits
        self.supplied_inputs = inputs
        self.supplied_externs = externs
        self.implicit_calibration = Calibration()
        self.scope = NamespaceNest()

        # TODO: Derive hardware qubits directly from supplied qubits
        if self.dsl_qubits is not None:
            for qubit_name in self.dsl_qubits:
                if qubit_name.startswith("$"):
                    self.scope.current.declare_qubit(qubit_name)

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

    def _qubit_name_from_frame(self, frame):
        signal_path = self.gate_store.ports[frame.port]
        for name, qubit in self.dsl_qubits.items():
            if signal_path in qubit.signals.values():
                return name

    def _resolved_qubit_names(self, qubits_or_frames):
        if all(isinstance(f, Frame) for f in qubits_or_frames):
            names = [self._qubit_name_from_frame(frame) for frame in qubits_or_frames]
        elif any(isinstance(f, Frame) for f in qubits_or_frames):
            msg = "Cannot mix frames and qubits."
            raise OpenQasmException(msg)
        else:
            names = [q.canonical_name for q in qubits_or_frames]
        return names

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
        elif isinstance(
            parent, (ast.Box, ast.CalibrationStatement, ast.CalibrationDefinition)
        ):
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
                elif isinstance(child, ast.IODeclaration):
                    self._handle_io_declaration(child)
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
                elif isinstance(child, ast.CalibrationDefinition):
                    self._handle_calibration_definition(child)
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
                    subsect = self._handle_assignment(child)
                elif isinstance(child, ast.QuantumMeasurementStatement):
                    subsect = self._handle_measurement(child)
                elif isinstance(child, ast.QuantumReset):
                    subsect = self._handle_quantum_reset(child)
                else:
                    msg = f"Statement type {type(child)} not supported"
                    raise OpenQasmException(msg, mark=child.span)
            except OpenQasmException as e:
                if e.mark is None:
                    e.mark = child.span
                raise
            except Exception as e:
                msg = f"Failed to process statement: {e!r}"
                raise OpenQasmException(msg, mark=child.span) from e
            if subsect is not None:
                sect.add(subsect)

        return sect

    def _handle_qubit_declaration(self, statement: ast.QubitDeclaration) -> None:
        name = statement.qubit.name
        try:
            if statement.size is not None:
                try:
                    size = eval_expression(
                        statement.size, namespace=self.scope, type_=int
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
                    type_=int,
                )
            else:
                value = None
            size = statement.type.size
            if size is not None:
                size = eval_expression(size, namespace=self.scope, type_=int)

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
            freq = eval_expression(
                statement.init_expression.arguments[1],
                namespace=self.scope,
                type_=(float, int, SweepParameter),
            )
            phase = eval_expression(
                statement.init_expression.arguments[2],
                namespace=self.scope,
                type_=(float, SweepParameter),
            )
            self.scope.current.declare_frame(name, port, freq, phase)
        elif isinstance(statement.type, ast.WaveformType):
            value = eval_expression(
                statement.init_expression,
                namespace=self.scope,
                # TODO: type_=waveform-type,
            )
            # TODO: The waveform should be registered on the current scope
            #       so that it can be manipulated as a varialble, but
            #       that requires adding support for a waveform type in
            #       namespaces and adding support in play.
            self.gate_store.register_waveform(name, value)
            self.scope.current.declare_classical_value(name, value)
        else:
            value = eval_expression(statement.init_expression, namespace=self.scope)
            self.scope.current.declare_classical_value(name, value)

    def _handle_io_declaration(self, statement: ast.IODeclaration):
        # The openqasm parse itself checks that IODeclarations
        # are only allowed at the top level scope. We assert
        # here to ensure our own code has not messed up:
        assert self.scope.current.toplevel
        if statement.io_identifier == ast.IOKeyword.output:
            raise OpenQasmException(
                "Output declarations are not yet supported by"
                " LabOne Q's OpenQASM 3 compiler."
            )
        elif statement.io_identifier == ast.IOKeyword.input:
            # TODO: Handle statement.type
            # TODO: Handle implicit inputs. If the input keyword
            #       is never used, the OpenQASM program way
            #       accept inputs for undefined variables implicitly.
            #       We can add support this by pre-walking the AST
            #       to check if the input keyword is used before
            #       doing the actual parsing. This requires adding
            #       and analysis step before compilation proper starts.
            name = statement.identifier.name
            if name in self.scope.current.local_scope:
                raise OpenQasmException(f"Re-declaration of input {name}")
            if self.supplied_inputs is None or name not in self.supplied_inputs:
                # TODO: This case can be removed once variables are
                #       properly supported by the compiler.
                raise OpenQasmException(f"Missing input {name}")
            self.scope.current.declare_classical_value(
                name, value=self.supplied_inputs[name]
            )
        else:
            raise OpenQasmException(
                f"Invalid IO direction identifier {statement.io_identifier}"
            )

    def _handle_constant_declaration(self, statement: ast.ConstantDeclaration) -> None:
        name = statement.identifier.name
        value = eval_expression(statement.init_expression, namespace=self.scope)
        self.scope.current.declare_classical_value(name, value)

    def _handle_extern_declaration(self, statement: ast.ExternDeclaration) -> None:
        # TODO: currently unused for 'extern port' due to _workaround_extern_port.
        name = statement.name.name
        # TODO: check that the scope is suitably top-level or
        #       delegate this check to declare_function
        if name not in self.supplied_externs:
            raise OpenQasmException("Extern function {name!r} not provided.")
        func = self.supplied_externs[name]
        arguments = statement.arguments
        return_type = statement.return_type
        self.scope.current.declare_function(name, arguments, return_type, func)

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
        qubits_or_frames = [
            eval_expression(qubit, namespace=self.scope) for qubit in statement.qubits
        ]
        qubit_names = self._resolved_qubit_names(qubits_or_frames)
        reserved_qubits = [self.dsl_qubits[name] for name in qubit_names]
        if not reserved_qubits:
            reserved_qubits = self.dsl_qubits.values()  # reserve all qubits

        sect = Section(uid=id_generator("barrier"))
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

    def _handle_calibration_definition(
        self, statement: ast.CalibrationDefinition
    ) -> None:
        defcal_name = statement.name.name
        # TODO: Add support for defcals that return values
        if statement.return_type is not None:
            raise OpenQasmException("defcal with return not yet supported")
        # TODO: Add support for defcals with arguments
        if statement.arguments:
            raise OpenQasmException("defcal with arguments not yet supported")

        qubit_names = tuple(q.name for q in statement.qubits)
        # TODO: Add support for placeholder qubits, possibly by just registering
        #       the gate for all hardware qubits.
        if any(not name.startswith("$") for name in qubit_names):
            raise OpenQasmException(
                "defcal statements for arbitrary qubits not yet supported"
            )

        try:
            section = self.transpile(statement, uid_hint="defcal")
        except OpenQasmException as e:
            # Spans on exceptions from inside cal and defcal blocks
            # are relative to the cal block and not the original qasm
            # source so we need to update them here:
            prefix_length = len(f"defcal {defcal_name} {' '.join(qubit_names)} {{")
            e.mark = _convert_openpulse_span(statement.span, e.mark, prefix_length)
            raise

        reserved_qubits = [self.dsl_qubits[qubit] for qubit in qubit_names]
        reserved_signals = set()
        for qubit in reserved_qubits:
            for exp_signal in qubit.experiment_signals():
                reserved_signals.add(exp_signal.mapped_logical_signal_path)
        for signal in reserved_signals:
            section.reserve(signal)

        def gate_factory(**kwargs):
            # TODO: The **kw above swallows the handle argument passed to
            #       measurement.
            return section

        self.gate_store.register_gate_section(defcal_name, qubit_names, gate_factory)

    def _handle_calibration(self, statement: ast.CalibrationStatement):
        try:
            return self.transpile(statement, uid_hint="calibration")
        except OpenQasmException as e:
            # Spans on exceptions from inside cal and defcal blocks
            # are relative to the cal block and not the original qasm
            # source so we need to update them here:
            prefix_length = len("cal {")
            e.mark = _convert_openpulse_span(statement.span, e.mark, prefix_length)
            raise

    def _handle_cal_expression(self, statement: ast.ExpressionStatement):
        expr = statement.expression
        if isinstance(expr, ast.FunctionCall):
            name = expr.name.name
            if name == "play":
                return self._handle_play(expr)
            elif name == "set_frequency":
                self._handle_set_frequency(expr)
            elif name == "shift_frequency":
                msg = "Not yet implemented: shift_frequency"
                OpenQasmException(msg, mark=statement.span)
            else:
                result = eval_expression(expr, namespace=self.scope)
                if isinstance(result, ExternResult):
                    # ignore the other value attributes since the
                    # result isn't assigned
                    section = result.section
                else:
                    section = None
                return section
        else:
            msg = (
                "Currently only function calls are supported as calibration"
                " expression statements."
            )
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

    def _handle_set_frequency(self, expr: ast.FunctionCall):
        assert len(expr.arguments) == 2
        frame = eval_expression(expr.arguments[0], namespace=self.scope)
        signal = self.gate_store.ports[frame.port]
        freq = eval_expression(
            expr.arguments[1],
            namespace=self.scope,
            type_=(float, int, SweepParameter),
        )

        # TODO: this overwrites the frequency for the whole experiment. We should
        #  instead update the existing frequency from this point in time on.
        self.implicit_calibration[signal] = SignalCalibration(
            oscillator=Oscillator(
                id_generator(f"osc_{frame.canonical_name}"),
                frequency=freq,
                modulation_type=ModulationType.HARDWARE,
            )
        )

    def _handle_delay_instruction(self, statement: ast.DelayInstruction):
        duration = eval_expression(
            statement.duration,
            namespace=self.scope,
            type_=(float, SweepParameter),
        )
        qubits_or_frames = [
            eval_expression(qubit, namespace=self.scope) for qubit in statement.qubits
        ]
        # OpenPulse allows for delaying only some of a qubit's signals
        selective_frame_delay = False
        if all(isinstance(f, Frame) for f in qubits_or_frames):
            selective_frame_delay = True
        qubit_names = self._resolved_qubit_names(qubits_or_frames)
        qubits_str = "_".join(qubit_names)
        delay_section = Section(uid=id_generator(f"{qubits_str}_delay"))
        for qubit in qubit_names:
            dsl_qubit = self.dsl_qubits[qubit]
            # TODO: I think we might be able to remove this loop with
            #       the new qubit class.
            # TODO: Is it correct to delay on only one line?
            # TODO: (convention was for regular pulse sheets, but inconsistent with spectroscopy)
            # TODO: What should happen for custom qubit types?
            if selective_frame_delay:
                for frame in qubits_or_frames:
                    delay_section.delay(
                        self.gate_store.ports[frame.port], time=duration
                    )
            else:
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
            start = eval_expression(
                loop_set_decl.start, namespace=self.scope, type_=int
            )
            stop = eval_expression(loop_set_decl.end, namespace=self.scope, type_=int)
            if loop_set_decl.step is not None:
                step = eval_expression(
                    loop_set_decl.step, namespace=self.scope, type_=int
                )
            else:
                step = 1
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
        if isinstance(rvalue, ExternResult):
            section = rvalue.section
            if rvalue.handle is not None:
                rvalue = MeasurementResult(handle=rvalue.handle)
            else:
                rvalue = rvalue.result
        else:
            section = None
        lvalue.value = op(lvalue.value, rvalue)
        return section

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
            b.value = MeasurementResult(handle=handle_name)
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
    inputs: dict[str, Any] | None = None,
    externs: dict[str, Callable] | None = None,
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
        inputs:
            Inputs to the OpenQASM program.
        externs:
            Extern functions for the OpenQASM program.
        count:
            The number of acquire iterations.
        averaging_mode:
            The mode of how to average the acquired data.
        acquisition_type:
            The type of acquisition to perform.
    """
    importer = OpenQasm3Importer(
        qubits=qubits, inputs=inputs, externs=externs, gate_store=gate_store
    )
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

    exp.set_calibration(importer.implicit_calibration)

    return exp
