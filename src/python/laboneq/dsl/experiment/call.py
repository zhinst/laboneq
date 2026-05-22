# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Callable, Union

import attrs

from laboneq.core.utilities.dsl_dataclass_decorator import classformatter

from .operation import Operation


@classformatter
@attrs.define(init=False)
class Call(Operation):
    """Class abstracting a function call."""

    func_name: Union[str, Callable] = attrs.field(default=None)
    args: dict[str, Any] = attrs.field(default=None)

    def __init__(self, func_name: Union[str, Callable], **kwargs):
        """Constructor.

        Args:
            func_name: Function that should be called.
            **kwargs (dict): Arguments of the function call.
        """
        self.func_name = func_name.__name__ if callable(func_name) else func_name
        self.args = kwargs
