# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import math
import operator

from openqasm3 import ast

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


def eval_expression(expression: ast.Expression):
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
        lhs = eval_expression(expression.lhs)
        rhs = eval_expression(expression.rhs)
        return op(lhs, rhs)
    if isinstance(expression, ast.UnaryExpression):
        try:
            op = unary_ops[expression.op.name]
        except KeyError as e:
            raise OpenQasmException(
                f"Unsupported operator '{expression.op}'", expression.span
            ) from e
        return op(eval_expression(expression.expression))

    if isinstance(expression, ast.Identifier):
        try:
            return constants[expression.name]
        except KeyError as e:
            raise OpenQasmException(
                f"Unknown identifier '{expression.name}'", expression.span
            ) from e

    raise OpenQasmException("Failed to evaluate expression", expression.span)
