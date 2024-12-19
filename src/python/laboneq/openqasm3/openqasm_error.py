# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from openqasm3.ast import Span


class OpenQasmException(Exception):
    """An exception class for OpenQASM related errors in LabOne Q.

    The exception can highlight the issue in the source text.

    If the `source` or the `mark` are not provided, it defaults to the usual behaviour of
    just printing the message.

    ```python
    Unknown identifier 'frame0'
    |          play(frame0, 10ns);
               ^
    ```
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
