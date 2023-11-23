# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from openqasm3.ast import Span


class OpenQasmException(Exception):
    """Exception that can properly highlight the issue in the source text

    >>> with open(__file__, "r") as f:
    ...    src = f.read()
    >>> mark = Span(9, 7-1, 9, 24-1)
    >>> raise OpenQasmException("Highlight the name of the class!", mark, src)  #doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    openqasm_error.OpenQasmException: Highlight the name of the class!
    |  class OpenQasmException(Exception):
             ^^^^^^^^^^^^^^^^^

    If the source or the mark are not provided, it defaults to the usual behaviour of
    just printing the message.

    >>> raise OpenQasmException("Something went wrong")
    Traceback (most recent call last):
      ...
    openqasm_error.OpenQasmException: Something went wrong
    """

    def __init__(self, msg=None, mark: Span | None = None, source: str | None = None):
        self.mark = mark
        self.source = source
        super().__init__(msg)

    def __str__(self):
        if self.source is None or self.mark is None:
            return super().__str__()

        msg = super().__str__()

        lines = self.source.splitlines()[self.mark.start_line - 1 : self.mark.end_line]
        if len(lines) > 1:
            marked_src = "\n".join(f"|  {l}" for l in lines)
        else:
            marked_src = (
                f"|  {lines[0]}\n"
                + "   "
                + self.mark.start_column * " "
                + max(1, self.mark.end_column - self.mark.start_column) * "^"
            )
        return msg + "\n" + marked_src
