# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.quantum.quantum_operations import QuantumOperations
from laboneq.openqasm3.openqasm_error import OpenQasmException
import openqasm3.visitor
from openqasm3.ast import QASMNode

import copy
import math
import operator
import re
from typing import Any, Callable, Union, TYPE_CHECKING
from laboneq.openqasm3 import namespace
from openpulse import ast
from laboneq._utils import id_generator
from laboneq.dsl import LinearSweepParameter, Parameter, SweepParameter
from laboneq.dsl.calibration import Calibration, Oscillator, SignalCalibration
from laboneq.dsl.enums import (
    AcquisitionType,
    ModulationType,
    SectionAlignment,
)
from laboneq.dsl.experiment import Section, Sweep
from laboneq.dsl.quantum.quantum_element import QuantumElement
from laboneq.openqasm3.expression import eval_expression, eval_lvalue
from laboneq.openqasm3.namespace import (
    Array,
    ClassicalRef,
    Frame,
    NamespaceStack,
    QubitRef,
    Waveform,
)
from laboneq.openqasm3.results import (
    ExternResult,
    MeasurementResult,
    TranspileResult,
)
from laboneq.dsl import quantum


if TYPE_CHECKING:
    from laboneq.dsl.experiment.pulse import Pulse


class _DefaultOperations(quantum.QuantumOperations):
    """Default quantum operations for OpenQASM programs."""

    QUBIT_TYPES = quantum.QuantumElement

    @quantum.quantum_operation
    def barrier(self, qubit):
        """Barrier gate.

        Reserves all signals on qubits.
        """
        # QuantumOperations reserves the lines automatically when qubits are passed


def _unwrap_qubit_register(name: str, size: int) -> list[str]:
    """Unwrap a QASM qubit register into a list of single qubits."""
    # Qubit register has a special convention
    # Hopefully a better solution is added later
    return [f"{name}[{idx}]" for idx in range(size)]


def _convert_openpulse_span(base_span, relative_span, prefix_length):
    """Add a relative OpenPulse span to a base OpenQASM parser
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


class TranspilerVisitor(openqasm3.visitor.QASMVisitor):
    def __init__(
        self,
        qops: QuantumOperations | None = None,
        qubits: dict[str, QuantumElement] | None = None,
        inputs: dict[str, Any] | None = None,
        externs: dict[str, Callable | str] | None = None,
        namespace: None | NamespaceStack = None,
    ):
        self.qubits = qubits
        # Default operations used when supplied `qops` does not implement them.
        self._default_ops = _DefaultOperations()
        # NOTE: The workaround of extern port declaration in the importer
        # and declaration of hardware qubits inject info into the namespace.
        # When these workaround are removed, the namespace should be created internally.
        self.namespace = (
            NamespaceStack() if namespace is None else copy.deepcopy(namespace)
        )
        self.acquire_loop_options = {}
        self.implicit_calibration = Calibration()
        self.qops = qops
        self.supplied_inputs = inputs or {}
        self.supplied_externs = externs or {}
        self._program_gates = {}

    def _register_gate_section(
        self, name: str, qubit_names: list[str], section_factory: Callable[..., Section]
    ):
        self._program_gates[(name, tuple(qubit_names))] = section_factory

    def _retrieve_gate(self, loc: ast.Span, name: str, qubits: tuple[str, list[str]]):
        """
        Retrieve a gate from the default operations.
        """
        try:
            if self.qops and name in self.qops:
                return self.qops[name]
            return self._default_ops[name]
        except KeyError:
            msg = f"Quantum operation '{name}' for qubit(s) {qubits} not found."
            raise OpenQasmException(msg, mark=loc) from None

    def _call_gate(
        self,
        loc: ast.Span,
        name: str,
        qubits: list[str | tuple[str]],
        args=None,
        kwargs=None,
    ) -> Section:
        """
        Returns the section implementing the given gate.

        When a gate is broadcast across multiple qubits, the underlying quantum
        operation returns multiple sections (one for each element of the broadcast)
        and this method wraps those sections in a single section to ensure correct
        timing behaviour.

        When a gate is not broadcast, this method returns the single section returned
        by the quantum operation.
        """
        args = args or ()
        kwargs = kwargs or {}
        qubit_args = []
        broadcast = False
        try:
            for x in qubits:
                if isinstance(x, (list, tuple)):
                    qubit_args.append([self.qubits[qubit] for qubit in x])
                    broadcast = True
                else:
                    qubit_args.append(self.qubits[x])
        except KeyError:
            err_msg = f"Qubit(s) {qubits} not supplied."
            raise OpenQasmException(err_msg, mark=loc) from None
        if not broadcast:
            if (name, tuple(qubits)) in self._program_gates:
                return self._program_gates[(name, tuple(qubits))](*args, **kwargs)
        gate_callable = self._retrieve_gate(loc, name, qubits)

        secs = gate_callable(*qubit_args, *args, **kwargs)
        if isinstance(secs, list):
            return Section(uid=id_generator(f"{name}_broadcast"), children=secs)
        return secs

    def _has_frame(self, qubits_or_frames) -> bool:
        return any(isinstance(f, Frame) for f in qubits_or_frames)

    def _frame_to_qubit(self, qubits_or_frames) -> list[str]:
        if all(isinstance(f, Frame) for f in qubits_or_frames):
            return [
                self.namespace.lookup(frame.port).qubit for frame in qubits_or_frames
            ]
        elif any(isinstance(f, Frame) for f in qubits_or_frames):
            msg = "Cannot mix frames and qubits."
            raise OpenQasmException(msg)
        return None

    def generic_visit(self, node: QASMNode, context=None):
        raise OpenQasmException(
            f"Statement type {type(node)} not supported", mark=node.span
        )

    def visit_QubitDeclaration(self, node: QASMNode):
        return self._handle_qubit_declaration(node)

    def visit_ClassicalDeclaration(self, node: QASMNode):
        return self._handle_classical_declaration(node)

    def visit_IODeclaration(self, node: QASMNode):
        return self._handle_io_declaration(node)

    def visit_ConstantDeclaration(self, node: QASMNode):
        return self._handle_constant_declaration(node)

    def visit_ExternDeclaration(self, node: QASMNode):
        return self._handle_extern_declaration(node)

    def visit_AliasStatement(self, node: QASMNode):
        return self._handle_alias_statement(node)

    def visit_Include(self, node: QASMNode):
        return self._handle_include(node)

    def visit_CalibrationGrammarDeclaration(self, node: QASMNode):
        return self._handle_calibration_grammar(node)

    def visit_CalibrationDefinition(self, node: QASMNode):
        return self._handle_calibration_definition(node)

    def visit_CalibrationStatement(self, node: QASMNode):
        return self._handle_calibration(node)

    def visit_ExpressionStatement(self, node: QASMNode):
        return self._handle_cal_expression(node)

    def visit_QuantumGate(self, node: QASMNode):
        return self._handle_quantum_gate(node)

    def visit_Box(self, node: QASMNode):
        return self._handle_box(node)

    def visit_QuantumBarrier(self, node: QASMNode):
        return self._handle_barrier(node)

    def visit_DelayInstruction(self, node: QASMNode):
        return self._handle_delay_instruction(node)

    def visit_BranchingStatement(self, node: QASMNode):
        return self._handle_branching_statement(node)

    def visit_ForInLoop(self, node: QASMNode):
        return self._handle_for_in_loop(node)

    def visit_ClassicalAssignment(self, node: QASMNode):
        return self._handle_assignment(node)

    def visit_QuantumMeasurementStatement(self, node: QASMNode):
        return self._handle_measurement(node)

    def visit_QuantumReset(self, node: QASMNode):
        return self._handle_quantum_reset(node)

    def visit_Pragma(self, node: QASMNode):
        return self._handle_pragma(node)

    def _transpile(
        self,
        parent: Union[ast.Program, ast.Box, ast.ForInLoop],
        uid_hint="",
    ) -> Section | None:
        """Transpile an OpenQASM construct into a section.

        Returns:
            Built Section or None in case of no-operations.
        """
        if isinstance(parent, ast.Program):
            body = parent.statements
        elif isinstance(
            parent,
            (ast.Box, ast.CalibrationStatement, ast.CalibrationDefinition),
        ):
            body = parent.body
        elif isinstance(parent, ast.ForInLoop):
            body = parent.block
        else:
            msg = f"Unsupported block type {type(parent)!r}"
            raise OpenQasmException(msg, mark=parent.span)
        sect_children = []
        try:
            for child in body:
                subsect = self.visit(child)
                if isinstance(subsect, Section):
                    if subsect.children:
                        sect_children.append(subsect)
                elif isinstance(subsect, list):
                    sect_children.extend(subsect)
                elif subsect is not None:
                    raise RuntimeError(
                        "Handlers must return Section, a list of Section or None"
                    )

        except OpenQasmException as e:
            if e.mark is None:
                e.mark = child.span
            raise
        except Exception as e:
            msg = f"Failed to process statement: {e!r}"
            raise OpenQasmException(msg, mark=child.span) from e
        if sect_children:
            return Section(uid=id_generator(uid_hint), children=sect_children)
        return None

    def transpile(
        self,
        parent: Union[ast.Program, ast.Box, ast.ForInLoop],
        uid_hint="",
    ) -> TranspileResult:
        sect = self._transpile(parent, uid_hint=uid_hint) or Section(
            uid=id_generator(uid_hint)
        )
        return TranspileResult(
            sect,
            acquire_loop_options=self.acquire_loop_options,
            implicit_calibration=self.implicit_calibration,
            variables=self.namespace.current.local_scope,
        )

    def _handle_qubit_declaration(self, statement: ast.QubitDeclaration) -> None:
        name = statement.qubit.name
        try:
            if statement.size is not None:
                try:
                    size = eval_expression(
                        statement.size,
                        namespace=self.namespace,
                        type_=int,
                    )
                except Exception:
                    msg = "Qubit declaration size must evaluate to an integer."
                    raise OpenQasmException(msg, mark=statement.span) from None

                qubits = [QubitRef(q) for q in _unwrap_qubit_register(name, size)]
                self.namespace.current.declare_reference(name, qubits)
            else:
                self.namespace.current.declare_qubit(name)
        except ValueError as e:
            raise OpenQasmException(str(e), mark=statement.span) from e
        except OpenQasmException as e:
            e.mark = statement.span
            raise

    def _handle_classical_declaration(
        self,
        statement: ast.ClassicalDeclaration,
    ) -> None:
        name = statement.identifier.name
        if isinstance(statement.type, ast.BitType):
            if statement.init_expression is not None:
                value = eval_expression(
                    statement.init_expression,
                    namespace=self.namespace,
                    type_=int,
                )
            else:
                value = None
            size = statement.type.size
            if size is not None:
                size = eval_expression(size, namespace=self.namespace, type_=int)

                # declare the individual bits...
                bits = [
                    self.namespace.current.declare_classical_value(
                        f"{name}[{i}]",
                        value=bool((value >> i) & 1) if value is not None else None,
                    )
                    for i in range(size)
                ]
                # ... as well as a list aliasing them
                self.namespace.current.declare_reference(name, bits)
            else:
                self.namespace.current.declare_classical_value(name, value)
        elif isinstance(statement.type, ast.FrameType):
            init = statement.init_expression
            if not isinstance(init, ast.FunctionCall) or init.name.name != "newframe":
                msg = "Frame type initializer must be a 'newframe' function call."
                raise OpenQasmException(msg, mark=statement.span)
            name = statement.identifier.name
            freq = eval_expression(
                statement.init_expression.arguments[1],
                namespace=self.namespace,
                type_=(float, int, SweepParameter),
            )
            phase = eval_expression(
                statement.init_expression.arguments[2],
                namespace=self.namespace,
                type_=(float, SweepParameter),
            )
            port = self.namespace.lookup(statement.init_expression.arguments[0].name)
            self.namespace.current.declare_frame(name, port.canonical_name, freq, phase)
        elif isinstance(statement.type, ast.WaveformType):
            # waveforms can be declared only in cal blocks.
            value = eval_expression(
                statement.init_expression,
                namespace=self.namespace,
                # TODO: type_=waveform-type,
                # waveform w = extern_waveform(...) would return arbitrary type.
                # We need to handle this better.
            )
            self.namespace.current.declare_waveform(name, value)
        elif isinstance(statement.type, ast.PortType):
            try:
                signal_path = self.supplied_externs[name]
            except KeyError:
                msg = f"Port {name!r} not provided."
                raise OpenQasmException(msg) from None
            for qname, qubit in self.qubits.items():
                if signal_path in qubit.signals.values():
                    self.namespace.current.declare_port(name, qname, signal_path)
                    break
            else:
                msg = f"Port {name} maps to non-existed signal {signal_path}."
                raise ValueError(msg)
        else:
            # TODO: We should set a clear boundary on what types are supported here
            # else should raise an exception
            if statement.init_expression is not None:
                value = eval_expression(
                    statement.init_expression, namespace=self.namespace
                )
            else:
                value = None
            self.namespace.current.declare_classical_value(name, value)

    def _handle_io_declaration(self, statement: ast.IODeclaration):
        # The openqasm parse itself checks that IODeclarations
        # are only allowed at the top level scope. We assert
        # here to ensure our own code has not messed up:
        assert isinstance(self.namespace.current, namespace.TopLevelNamespace)
        if statement.io_identifier == ast.IOKeyword.output:
            raise OpenQasmException(
                "Output declarations are not yet supported by"
                " LabOne Q's OpenQASM 3 compiler.",
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
            if name in self.namespace.current.local_scope:
                raise OpenQasmException(f"Re-declaration of input {name}")
            if name not in self.supplied_inputs:
                # TODO: This case can be removed once variables are
                #       properly supported by the compiler.
                raise OpenQasmException(f"Missing input {name}")
            self.namespace.current.declare_classical_value(
                name,
                value=self.supplied_inputs[name],
            )
        else:
            raise OpenQasmException(
                f"Invalid IO direction identifier {statement.io_identifier}",
            )

    def _handle_constant_declaration(self, statement: ast.ConstantDeclaration) -> None:
        name = statement.identifier.name
        value = eval_expression(statement.init_expression, namespace=self.namespace)
        self.namespace.current.declare_classical_value(name, value)

    def _handle_extern_declaration(self, statement: ast.ExternDeclaration) -> None:
        # TODO: currently unused for 'extern port' due to _workaround_extern_port.
        name = statement.name.name
        # TODO: check that the scope is suitably top-level or
        #       delegate this check to declare_function
        if name not in self.supplied_externs:
            msg = f"Extern function {name!r} not provided."
            raise OpenQasmException(msg, mark=statement.span)
        func = self.supplied_externs[name]
        arguments = statement.arguments
        return_type = statement.return_type
        self.namespace.current.declare_function(name, arguments, return_type, func)

    def _handle_alias_statement(self, statement: ast.AliasStatement):
        if not isinstance(statement.target, ast.Identifier):
            msg = "Alias target must be an identifier."
            raise OpenQasmException(msg, mark=statement.span)
        name = statement.target.name

        try:
            value = eval_lvalue(statement.value, namespace=self.namespace)
        except OpenQasmException:
            raise
        except Exception as e:
            msg = "Invalid alias value"
            raise OpenQasmException(msg, mark=statement.value.span) from e
        try:
            self.namespace.current.declare_reference(name, value)
        except OpenQasmException as e:
            e.mark = statement.span
            raise

    def _process_qubit_register(self, qubits) -> tuple[str | list[str]]:
        qubit_names = []
        _qubit_register_length = None
        for qubit in qubits:
            if isinstance(qubit, list):
                # qubit register
                qubit_names.append(tuple([q.canonical_name for q in qubit]))
                if len(qubit) != _qubit_register_length:
                    if _qubit_register_length is None:
                        _qubit_register_length = len(qubit)
                    else:
                        msg = "Qubit registers must have the same length."
                        raise ValueError(msg)
            elif isinstance(qubit, QubitRef):
                qubit_names.append(qubit.canonical_name)
            else:
                raise TypeError(f"Qubit expected, got '{type(qubit).__name__}'")

        # TODO: Remove this restriction.
        # qubit_names must be a tuple to be hashable for retrieving program_gates.
        return tuple(qubit_names)

    def _handle_quantum_gate(self, statement: ast.QuantumGate):
        args = tuple(
            eval_expression(arg, namespace=self.namespace)
            for arg in statement.arguments
        )
        if statement.modifiers or statement.duration:
            msg = "Gate modifiers and duration not yet supported."
            raise OpenQasmException(msg, mark=statement.span)
        if not isinstance(statement.name, ast.Identifier):
            msg = "Gate name must be an identifier."
            raise OpenQasmException(msg, mark=statement.span)

        name = statement.name.name
        qubits = [
            eval_expression(q, namespace=self.namespace) for q in statement.qubits
        ]
        try:
            qubit_names = self._process_qubit_register(qubits)
        except (ValueError, TypeError) as e:
            raise OpenQasmException(str(e), mark=statement.span) from None
        return self._call_gate(
            statement.span,
            name,
            qubit_names,
            args=args,
        )

    def _handle_box(self, statement: ast.Box):
        if statement.duration:
            raise ValueError("Box duration not yet supported.")
        with self.namespace.new_scope():
            return self._transpile(statement, uid_hint="box")

    def _handle_barrier(self, statement: ast.QuantumBarrier):
        # If no arguments are given, reserve all qubits
        if not statement.qubits:
            qubits = self.qubits.keys()
        else:
            # barrier can be applied to either qubits or frames
            qubits_or_frames = [
                eval_expression(q, namespace=self.namespace) for q in statement.qubits
            ]
            if self._has_frame(qubits_or_frames):
                # Avoid duplicate qubits due to openpulse frames
                qubits = tuple(dict.fromkeys(self._frame_to_qubit(qubits_or_frames)))
            else:
                qubits = self._process_qubit_register(qubits_or_frames)
        # force barrier to be broadcasted if more than one qubit are given
        # or if a qubit register is given by putting all qubits in a list

        input_qubits = []
        if len(qubits) == 1:
            input_qubits = qubits
        else:
            for qubit in qubits:
                if isinstance(qubit, tuple):
                    input_qubits.extend(qubit)
                else:
                    input_qubits.append(qubit)
            input_qubits = [tuple(input_qubits)]
        return self._call_gate(
            statement.span,
            "barrier",
            input_qubits,
        )

    def _handle_include(self, statement: ast.Include) -> None:
        if statement.filename != "stdgates.inc":
            msg = f"Only 'stdgates.inc' is supported for include, found '{statement.filename}'."
            raise OpenQasmException(msg, mark=statement.span)

    def _handle_calibration_grammar(
        self,
        statement: ast.CalibrationGrammarDeclaration,
    ) -> None:
        if statement.name != "openpulse":
            msg = f"Only 'openpulse' is supported for defcalgrammar, found '{statement.name}'."
            raise OpenQasmException(msg, mark=statement.span)

    def _handle_calibration_definition(
        self,
        statement: ast.CalibrationDefinition,
    ) -> None:
        defcal_name = statement.name.name
        qubit_names = tuple(q.name for q in statement.qubits)

        def gate_factory(*args, **kwargs):
            with self.namespace.new_scope():
                resolved_args = {}
                for value, arg in zip(args, statement.arguments):
                    resolved_args[arg.name.name] = value

                # TODO: Add support for defcals that return values
                if statement.return_type is not None:
                    raise OpenQasmException("defcal with return not yet supported")

                if statement.arguments:
                    for arg in statement.arguments:
                        name = arg.name.name
                        self.namespace.current.declare_classical_value(
                            name,
                            resolved_args[name],
                        )

                # TODO: Add support for placeholder qubits, possibly by just registering
                #       the gate for all hardware qubits.
                if any(not name.startswith("$") for name in qubit_names):
                    raise OpenQasmException(
                        "defcal statements for arbitrary qubits not yet supported",
                    )

                try:
                    section = self.transpile(statement, uid_hint="defcal").section
                except OpenQasmException as e:
                    # Spans on exceptions from inside cal and defcal blocks
                    # are relative to the cal block and not the original qasm
                    # source so we need to update them here:
                    prefix_length = len(
                        f"defcal {defcal_name} {' '.join(qubit_names)} {{",
                    )
                    e.mark = _convert_openpulse_span(
                        statement.span,
                        e.mark,
                        prefix_length,
                    )
                    raise

                reserved_qubits = [self.qubits[qubit] for qubit in qubit_names]
                reserved_signals = set()
                for qubit in reserved_qubits:
                    for exp_signal in qubit.experiment_signals():
                        reserved_signals.add(exp_signal.mapped_logical_signal_path)
                for signal in reserved_signals:
                    section.reserve(signal)

                return section

        self._register_gate_section(defcal_name, qubit_names, gate_factory)

    def _handle_calibration(self, statement: ast.CalibrationStatement):
        try:
            return self._transpile(statement, uid_hint="calibration")
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
                result = eval_expression(expr, namespace=self.namespace)
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

    def _get_waveform(self, name: str) -> Pulse:
        # TODO: This method should not need to exist. Waveforms should be looked up
        #       in the same way as any other program input.

        # Check first program defined waveforms,
        # then extern global defined waveforms
        try:
            return self.namespace.lookup(name).pulse()
        except KeyError:
            if name in self.supplied_inputs:
                return Waveform(name, self.supplied_inputs[name]).pulse()
        msg = f"Waveform {name!r} not provided."
        raise OpenQasmException(msg)

    def _handle_play(self, expr: ast.FunctionCall):
        frame: Frame = eval_expression(expr.arguments[0], namespace=self.namespace)
        arg1 = expr.arguments[1]
        if isinstance(arg1, ast.FunctionCall):
            if arg1.name.name == "scale":
                waveform = arg1.arguments[1].name
                amplitude = eval_expression(arg1.arguments[0], namespace=self.namespace)
            else:
                msg = "Currently only 'scale' is supported as a play waveform modifier function."
                raise OpenQasmException(msg, mark=expr.span)
        else:
            waveform = arg1.name
            amplitude = None
        pulse = self._get_waveform(waveform)
        sect = Section(uid=id_generator(f"play_{frame.port}"))
        sect.play(
            signal=self.namespace.lookup(frame.port).value,
            pulse=pulse,
            amplitude=amplitude,
        )
        return sect

    def _handle_set_frequency(self, expr: ast.FunctionCall):
        assert len(expr.arguments) == 2
        frame = eval_expression(expr.arguments[0], namespace=self.namespace)
        signal = self.namespace.lookup(frame.port).value
        freq = eval_expression(
            expr.arguments[1],
            namespace=self.namespace,
            type_=(float, int, SweepParameter, ast.ClassicalArgument),
        )

        # TODO: This handles explicit multiple calls to set_frequency. As long as a
        #  single call in a for loop is not recogised as a frequency sweep, this will not trigger.
        if signal in self.implicit_calibration.calibration_items:
            msg = "Setting the frequency more than once on a given signal is not supported in the current implementation."
            raise OpenQasmException(msg, mark=expr.span)
        # TODO: this overwrites the frequency for the whole experiment. We should
        #  instead update the existing frequency from this point in time on.
        self.implicit_calibration[signal] = SignalCalibration(
            oscillator=Oscillator(
                id_generator(f"osc_{frame.canonical_name}"),
                frequency=freq,
                modulation_type=ModulationType.HARDWARE,
            ),
        )

    def _handle_delay_instruction(self, statement: ast.DelayInstruction):
        duration = eval_expression(
            statement.duration,
            namespace=self.namespace,
            type_=(float, int, SweepParameter, ast.ClassicalArgument),
        )
        qubits_or_frames = [
            eval_expression(qubit, namespace=self.namespace)
            for qubit in statement.qubits
        ]
        # OpenPulse allows for delaying only some of a qubit's signals
        selective_frame_delay = False

        if self._has_frame(qubits_or_frames):
            # Avoid duplicate qubits due to openpulse frames
            selective_frame_delay = True
            qubit_names = self._frame_to_qubit(qubits_or_frames)
        else:
            nested_qubit_names = self._process_qubit_register(qubits_or_frames)
            # no broadcasting for delay operation
            # so we need to flatten the qubit names
            qubit_names = []
            for qubit in nested_qubit_names:
                if isinstance(qubit, (list, tuple)):
                    qubit_names.extend(qubit)
                else:
                    qubit_names.append(qubit)
        qubit_names = tuple(dict.fromkeys(qubit_names))

        qubits_str = "_".join(qubit_names)
        delay_section = Section(uid=id_generator(f"{qubits_str}_delay"))
        for qubit in qubit_names:
            dsl_qubit = self.qubits[qubit]
            # TODO: I think we might be able to remove this loop with
            #       the new qubit class.
            # TODO: Is it correct to delay on only one line?
            # TODO: (convention was for regular pulse sheets, but inconsistent with spectroscopy)
            # TODO: What should happen for custom qubit types?
            if selective_frame_delay:
                for frame in qubits_or_frames:
                    delay_section.delay(
                        signal=self.namespace.lookup(frame.port).value,
                        time=duration,
                    )
            else:
                # TODO: Use the quantum operation delay
                if "drive" in dsl_qubit.signals:
                    delay_section.delay(dsl_qubit.signals["drive"], time=duration)
        if not delay_section.children:
            msg = (
                f"Unable to apply delay to {qubit_names} due to missing drive signals."
            )
            raise OpenQasmException(msg, mark=statement.span)

        return delay_section

    def _handle_branching_statement(self, statement: ast.BranchingStatement):
        condition = eval_expression(statement.condition, namespace=self.namespace)
        if_block = None
        if statement.if_block:
            if_block = ast.Box(body=statement.if_block, duration=None)
        else_block = None
        if statement.else_block:
            else_block = ast.Box(body=statement.else_block, duration=None)

        if isinstance(condition, Parameter):
            raise OpenQasmException(
                "Branching on a sweep parameter is not"
                " yet supported by the LabOne Q OpenQASM importer.",
                mark=statement.condition.span,
            )

        if isinstance(condition, MeasurementResult):
            raise OpenQasmException(
                "Branching on a measurement result is not"
                " yet supported by the LabOne Q OpenQASM importer.",
                mark=statement.condition.span,
            )

        if not isinstance(condition, (int, float)):
            raise OpenQasmException(
                f"OpenQASM if conditions must be castable to bool."
                f" Got {type(condition).__name__} {condition!r} instead.",
                mark=statement.condition.span,
            )

        if condition:
            if if_block:
                with self.namespace.new_scope():
                    return self._transpile(if_block, uid_hint="if_block")
        else:
            if else_block:
                with self.namespace.new_scope():
                    return self._transpile(else_block, uid_hint="else_block")

    def _handle_for_in_loop(self, statement: ast.ForInLoop):
        loop_var = statement.identifier.name
        loop_set_decl = statement.set_declaration

        if isinstance(loop_set_decl, ast.RangeDefinition):
            start = eval_expression(
                loop_set_decl.start,
                namespace=self.namespace,
                type_=int,
            )
            stop = eval_expression(
                loop_set_decl.end, namespace=self.namespace, type_=int
            )
            if loop_set_decl.step is not None:
                step = eval_expression(
                    loop_set_decl.step,
                    namespace=self.namespace,
                    type_=int,
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

        with self.namespace.new_scope():
            self.namespace.current.declare_classical_value(loop_var, sweep_param)
            subsect = self._transpile(statement, uid_hint="block")
            if isinstance(subsect, Section):
                sweep.add(subsect)

        return sweep

    def _handle_assignment(self, statement: ast.ClassicalAssignment):
        lvalue = eval_lvalue(statement.lvalue, namespace=self.namespace)
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
        rvalue = eval_expression(statement.rvalue, namespace=self.namespace)
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
        qubits = eval_expression(statement.measure.qubit, namespace=self.namespace)
        bits = statement.target
        if bits is None:
            raise OpenQasmException(
                "Measurement must be assigned to a classical bit",
                mark=statement.span,
            )
        bits = eval_lvalue(statement.target, namespace=self.namespace)
        if isinstance(qubits, list):
            err_msg = None
            if not isinstance(bits, Array):
                err_msg = "Both bits and qubits must be either scalar or registers."
            bits = bits.value
            if len(bits) != len(qubits):
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
            section = self._call_gate(
                statement.span,
                "measure",
                (qubit_name,),
                kwargs={"handle": handle_name},
            )
            s.add(section)
            # Set the bit to a special value to disallow compile time arithmetic
            b.value = MeasurementResult(handle=handle_name)
        return s

    def _handle_quantum_reset(self, statement: ast.QuantumReset):
        # Although ``qubits`` is plural, only a single qubit is allowed.
        qubit_name = eval_expression(
            statement.qubits,
            namespace=self.namespace,
        ).canonical_name
        return self._call_gate(statement.span, "reset", (qubit_name,))

    _PRAGMA_ZI_PREFIX = "zi."

    _PRAGMA_ZI_STATEMENTS_RE = re.compile(
        r"""
        # ORed list of supported statements:
        (zi\.acquisition_type[ \t]+(?P<acquisition_type>[^ \t]*))
    """,
        re.VERBOSE,
    )

    def _handle_pragma(self, statement: ast.Pragma):
        pragma = statement.command

        if not pragma.startswith(self._PRAGMA_ZI_PREFIX):
            # we only process pragmas marked for Zurich Instruments
            return

        match = self._PRAGMA_ZI_STATEMENTS_RE.fullmatch(pragma)
        if match is None:
            msg = f"Invalid Zurich Instruments (zi.) pragma body: {pragma!r}"
            raise OpenQasmException(msg)
        if acquisition_type := match.group("acquisition_type"):
            return self._pragma_acquisition_type(acquisition_type)
        # The RuntimeError below should be unreachable -- it is a sanity
        # check to guard against cases from _PRAGMA_ZI_STATEMENTS_RE not being
        # handled in the lines above:
        raise RuntimeError(f"Unknown zhinst.com pragma statement: {pragma!r}")

    def _pragma_acquisition_type(self, acquisition_type: str):
        """Set the acquisition type specified via a pragma."""
        try:
            acquisition_type = AcquisitionType[acquisition_type.upper()]
        except Exception:
            msg = f"Invalid acquisition type {acquisition_type!r}"
            raise OpenQasmException(msg) from None
        if existing_type := self.acquire_loop_options.get("acquisition_type"):
            if existing_type != acquisition_type:
                msg = f"Attempt to change acquisition_type from {existing_type!r} to {acquisition_type!r}"
                raise OpenQasmException(msg)
        self.acquire_loop_options["acquisition_type"] = acquisition_type
