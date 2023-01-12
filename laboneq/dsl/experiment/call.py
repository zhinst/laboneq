# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Union

from .operation import Operation


@dataclass(init=False, repr=True, order=True)
class Call(Operation):
    """Class abstracting a function call."""

    func_name: Union[str, Callable] = field(default=None)
    args: Dict[str, Any] = field(default=None)

    def __init__(self, func_name: Union[str, Callable], **kwargs):
        """Constructor

        Args:
            func_name: Function that should be called.
            kwargs: Arguments of the function call.
        """
        self.func_name = func_name.__name__ if callable(func_name) else func_name
        self.args = {k: v.uid if hasattr(v, "uid") else v for k, v in kwargs.items()}
