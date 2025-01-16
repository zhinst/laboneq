# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from laboneq.dsl.experiment.section import Section
from laboneq.openqasm3.openqasm_error import OpenQasmException
from laboneq.dsl.calibration import Calibration


class MeasurementResult:
    """An internal holder for measurement results."""

    def __init__(self, handle: str):
        self.handle = handle


class ExternResult:
    """A class for holding the result of an extern function that
    needs to play a section, perform a measurement or other operations in
    addition to returning a result.

    Arguments:
    ---------
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
    ----------
        result (Any):
            The result returned by the extern function.
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
                f", not both: {self!r}",
            )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__}"
            f" result={self.result!r}"
            f" handle={self.handle!r}"
            f" has_section={self.section is not None}>"
        )


@dataclass(frozen=True)
class TranspileResult:
    """A collection of the results of transpiling a QASM program.

    Attributes:
        section:
            LabOneQ section.
        acquire_loop_options:
            Acquire loop options set in zi pragma.
        implicit_calibration:
            Implicit calibration set in zi pragma.
        variables:
            Global variables defined in the QASM program.

    """

    section: Section
    acquire_loop_options: dict
    implicit_calibration: Calibration
    variables: dict
