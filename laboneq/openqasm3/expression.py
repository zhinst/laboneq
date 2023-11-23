# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import operator
from typing import Iterable, Type

from openqasm3 import ast

from .namespace import ClassicalRef, Frame, NamespaceNest, QubitRef
from .openqasm_error import OpenQasmException

binary_ops = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    # todo: add other operators as required
}

unary_ops = {
    "-": operator.neg,
    # todo: add other operators as required
}

constants = {
    "pi": math.pi,
    "euler": math.e,
    "tau": math.tau,
    "π": math.pi,
    "τ": math.tau,
    "ℇ": math.e,
}


def duration_to_seconds(duration: ast.DurationLiteral):
    val = duration.value
    scale = {
        ast.TimeUnit.s: 1,
        ast.TimeUnit.ms: 1e-3,
        ast.TimeUnit.us: 1e-6,
        ast.TimeUnit.ns: 1e-9,
    }

    if duration.unit == ast.TimeUnit.dt:
        raise OpenQasmException("Backend-dependent duration not supported")
    return val * scale[duration.unit]


def _eval_expression(
    expression: ast.Expression | ast.DiscreteSet | None, namespace: NamespaceNest
):
    if expression is None:
        return None
    if isinstance(
        expression,
        (
            ast.IntegerLiteral,
            ast.FloatLiteral,
            ast.BooleanLiteral,
            ast.BitstringLiteral,
        ),
    ):
        return expression.value
    if isinstance(expression, ast.ImaginaryLiteral):
        return 1j * expression.value
    if isinstance(expression, ast.DurationLiteral):
        return duration_to_seconds(expression)
    if isinstance(expression, ast.BinaryExpression):
        try:
            op = binary_ops[expression.op.name]
        except KeyError as e:
            raise OpenQasmException(
                f"Unsupported operator '{expression.op.name}'", expression.span
            ) from e
        lhs = _eval_expression(expression.lhs, namespace)
        rhs = _eval_expression(expression.rhs, namespace)
        return op(lhs, rhs)
    if isinstance(expression, ast.UnaryExpression):
        try:
            op = unary_ops[expression.op.name]
        except KeyError as e:
            raise OpenQasmException(
                f"Unsupported operator '{expression.op}'", expression.span
            ) from e
        return op(_eval_expression(expression.expression, namespace))

    if isinstance(expression, ast.Identifier):
        try:
            return constants[expression.name]
        except KeyError:
            pass
        value = eval_lvalue(expression, namespace)
        if isinstance(value, QubitRef):
            return value
        if isinstance(value, ClassicalRef):
            # in an expression (rvalue), we can safely dereference classical values
            return value.value
        if isinstance(value, Frame):
            return value
        else:  # value is a list; dereference classical references
            return [v.value if isinstance(v, ClassicalRef) else v for v in value]
    if isinstance(expression, ast.RangeDefinition):
        start = _eval_expression(expression.start, namespace)
        end = _eval_expression(expression.end, namespace)
        end += 1 if end >= start else -1
        step = _eval_expression(expression.step, namespace) or 1
        return range(start, end, step)
    if isinstance(expression, ast.DiscreteSet):
        return [_eval_expression(x, namespace) for x in expression.values]

    if isinstance(expression, ast.IndexExpression):
        collection = _eval_expression(expression.collection, namespace)
        if not isinstance(collection, list):
            raise OpenQasmException("Collection expected", mark=expression.span)
        if isinstance(expression.index, list):
            indices = [_eval_expression(i, namespace) for i in expression.index]
        elif isinstance(expression.index, ast.DiscreteSet):
            indices = _eval_expression(expression.index, namespace)
        retval = []
        for index in indices:
            try:
                if isinstance(index, Iterable):
                    retval.extend([collection[i] for i in index])
                else:
                    retval.append(collection[index])
            except IndexError as e:  # noqa: PERF203
                raise OpenQasmException(str(e), mark=expression.span) from None
        if len(retval) == 1:
            retval = retval[0]
        return retval

    if isinstance(expression, ast.IndexedIdentifier):
        # IndexedIdentifier as an r-value
        collection = _eval_expression(expression.name, namespace)
        for index_element in expression.indices:
            if not isinstance(index_element, list) and not len(index_element) == 1:
                raise OpenQasmException("Unsupported index")
            index = _eval_expression(index_element[0], namespace)
            collection = collection[index]
        if isinstance(collection, list) and len(collection) == 1:
            return collection[0]
        return collection

    if isinstance(expression, ast.Concatenation):
        lhs = _eval_expression(expression.lhs, namespace)
        rhs = _eval_expression(expression.rhs, namespace)
        return [*lhs, *rhs]

    raise OpenQasmException(
        "Failed to evaluate expression", getattr(expression, "span", None)
    )


def eval_expression(
    expression: ast.Expression | None,
    *,
    namespace: NamespaceNest | None = None,
    type: Type | None = None,
):
    if namespace is None:
        namespace = NamespaceNest()
    try:
        retval = _eval_expression(expression, namespace)
    except OpenQasmException:
        raise
    except Exception as e:
        raise OpenQasmException(str(e), mark=expression.span) from e

    if type is not None and not isinstance(retval, type):
        raise OpenQasmException(
            f"Expected expression of type {type}, got {type(retval)}", expression.span
        )
    return retval


def eval_lvalue(
    node, namespace: NamespaceNest
) -> ClassicalRef | QubitRef | list[ClassicalRef] | list[QubitRef]:
    if isinstance(node, ast.Identifier):
        if node.name in constants:
            raise OpenQasmException("Cannot alias a constant", node.span)
        try:
            return namespace.lookup(node.name)
        except KeyError as e:
            raise OpenQasmException(
                f"Unknown identifier '{node.name}'", node.span
            ) from e
    if isinstance(node, ast.IndexedIdentifier):
        identifier = node.name
        if identifier.name in constants:
            raise OpenQasmException("Cannot alias a constant", node.span)
        collection = namespace.lookup(identifier.name)
        if isinstance(collection, QubitRef):
            raise OpenQasmException("Cannot index a single qubit", node.span)
        for index_element in node.indices:
            if isinstance(index_element, ast.DiscreteSet):
                index = _eval_expression(index_element, namespace)
            elif isinstance(index_element, list) and len(index_element) == 1:
                index_element = index_element[0]
                if isinstance(index_element, ast.RangeDefinition):
                    index = _eval_expression(index_element, namespace)
                else:
                    index = [_eval_expression(index_element, namespace)]
            else:
                raise OpenQasmException("Unsupported index")

            try:
                collection = [collection[i] for i in index]
            except Exception as e:
                raise OpenQasmException(f"Indexing failed: {str(e)}", node.span) from e

        if isinstance(collection, list) and len(collection) == 1:
            return collection[0]
        return collection

    if isinstance(node, ast.IndexExpression):
        # When aliasing an indexed identifier, e.g.:
        #    let foo = bar[1];
        # the RHS is currently parsed as an IndexExpression.
        # We'll be storing it as an lvalue so it is not really an expression...
        # We'll work around this by casting the syntax tree node to an IndexIdentifier
        # and go from there.
        if not isinstance(node.collection, ast.Identifier):
            raise OpenQasmException("Identifier expected", node.collection.span)
        new_node = ast.IndexedIdentifier(
            name=node.collection,
            indices=[node.index],
        )
        new_node.span = node.span
        return eval_lvalue(new_node, namespace)

    if isinstance(node, ast.Concatenation):
        lhs = eval_lvalue(node.lhs, namespace=namespace)
        rhs = eval_lvalue(node.rhs, namespace=namespace)
        for child, lval in ((node.lhs, lhs), (node.rhs, rhs)):
            if not isinstance(lval, list):
                raise OpenQasmException(
                    f"Array view expected, got {type(lval.__name__)}", child.span
                )
        return [*lhs, *rhs]
    raise OpenQasmException("lvalue expected", node.span)
