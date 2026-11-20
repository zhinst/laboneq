# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Parsing of OpenQASM 3 programs with OpenPulse calibration blocks.

Both backends build `openpulse.ast` trees.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING

import openpulse
from openpulse.parser import OpenPulseParsingError

if TYPE_CHECKING:
    from types import ModuleType

    from openpulse import ast

BACKEND_VARIABLE = "LABONEQ_OPENQASM_PARSER"

AUTO = "auto"
BUILTIN = "builtin"
OPENPULSE = "openpulse"
BACKENDS = (AUTO, BUILTIN, OPENPULSE)

RELATIVE_CALIBRATION_SPANS = "_laboneq_relative_calibration_spans"


def parse(text: str) -> ast.Program:
    """Parse an OpenQASM 3 program into an `openpulse.ast.Program`.

    `LABONEQ_OPENQASM_PARSER` selects the backend: `builtin`, `openpulse`, or
    `auto`, the default, which uses `builtin` wherever it is available.

    Raises:
        QASM3ParsingError: The program does not parse.
    """
    if selected_backend() == OPENPULSE:
        program = openpulse.parse(_extern_port_as_port(text))
        relative = True
    else:
        builtin = _builtin_parser()
        try:
            program = builtin.parse_openpulse(text)
        except builtin.ParseError as error:
            raise _openpulse_error(type(error), error) from None
        relative = False
    # A tree outlives the setting that produced it, and a caller may hand the
    # transpiler a tree this module never parsed, so each tree records which
    # convention it follows.
    setattr(program, RELATIVE_CALIBRATION_SPANS, relative)
    return program


def selected_backend() -> str:
    """Return the backend `parse` uses, `builtin` or `openpulse`."""
    requested = os.environ.get(BACKEND_VARIABLE, "").strip() or AUTO
    if requested not in BACKENDS:
        raise ValueError(
            f"{BACKEND_VARIABLE}={requested!r} is not one of {', '.join(BACKENDS)}"
        )
    if requested == OPENPULSE:
        return OPENPULSE
    if _builtin_parser() is not None:
        return BUILTIN
    if requested == BUILTIN:
        raise RuntimeError(
            f"{BACKEND_VARIABLE}={BUILTIN}, but this LabOne Q has no built-in parser. "
            "Install the wheel from PyPI, or unset the variable to use the openpulse "
            "package."
        )
    return OPENPULSE


def calibration_spans_are_relative(node: ast.QASMNode) -> bool:
    """Whether spans inside a `cal` body are reported against that body.

    `openpulse` reports them relative to the block, so a caller has to add the
    block's own position back. The built-in parser reports absolute spans. A tree
    that `parse` did not produce is taken to follow the `openpulse` convention.
    """
    return getattr(node, RELATIVE_CALIBRATION_SPANS, True)


def _extern_port_as_port(text: str) -> str:
    # openpulse 1.0.1 parses `port p;` but not `extern port p;`. Spaces in place
    # of `extern` keep every column where the source has it.
    # TODO: Delete once LabOne Q requires an openpulse release that parses
    #       'extern port'.
    return re.sub(r"\bextern(?=\s+port\b)", " " * len("extern"), text)


@lru_cache(maxsize=1)
def _builtin_parser() -> ModuleType | None:
    try:
        from laboneq._rust import openqasm3
    except ImportError:
        return None
    return openqasm3


@lru_cache(maxsize=1)
def _openpulse_error_class(base: type[Exception]) -> type[Exception]:
    """A `ParseError` that an `except OpenPulseParsingError` clause also catches.

    `OpenPulseParsingError` derives from `Exception` alone, so it is a sibling of
    the bindings' `ParseError` rather than a base, and has to be mixed in.
    """
    error_class = type(
        "ParseError",
        (base, OpenPulseParsingError),
        {"__module__": __name__, "__doc__": base.__doc__},
    )
    # pickle looks a class up by module and name, so the name has to resolve here.
    globals()[error_class.__name__] = error_class
    return error_class


def _openpulse_error(base: type[Exception], error: Exception) -> Exception:
    wrapped = _openpulse_error_class(base)(*error.args)
    wrapped.__dict__.update(vars(error))
    return wrapped
