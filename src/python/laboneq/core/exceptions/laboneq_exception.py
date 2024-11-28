# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


class LabOneQException(Exception):
    """Base class for exceptions raised by LabOne Q.

    Where appropriate, LabOne Q also raises built-in Python
    exceptions such as [ValueError][], [TypeError][] and [RuntimeError][].
    """
