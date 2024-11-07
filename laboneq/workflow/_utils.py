# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import inspect
from typing import Callable


def create_argument_map(func: Callable, *args: object, **kwargs: object) -> dict:
    """Create a mapping out of function arguments.

    Arguments:
        func: Callable
        *args: Arguments of the callable.
        **kwargs: Keyword arguments of the callable.

    Returns:
        An ordered dict of the arguments.
    """
    sig = inspect.signature(func)
    var_kw = None
    for arg, val in sig.parameters.items():
        if val.kind == inspect._ParameterKind.VAR_KEYWORD:
            var_kw = arg
            break
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    arg_map = dict(bound.arguments)
    if var_kw is not None:
        arg_map = arg_map | arg_map.pop(var_kw)
    return arg_map
