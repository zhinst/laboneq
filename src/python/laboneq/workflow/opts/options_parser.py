# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Options parser for workflows."""

from __future__ import annotations

import inspect
import sys
import typing

_PY_V39 = sys.version_info < (3, 10)
_PY_LT_V313 = sys.version_info < (3, 13)

if not _PY_V39:
    import types

_typing_eval_type = typing._eval_type
if _PY_LT_V313:
    # type_params arg is introduced in 3.13 and not supplying it is deprecated at the same time.

    def _eval_type(*args, type_params=None, **kwargs):
        return _typing_eval_type(*args, **kwargs)
else:
    _eval_type = _typing_eval_type


class TypeConvertError(Exception):
    """Exception raised for errors encountered during converting string to type."""


def _get_argument_types(
    fn: typing.Callable[..., object],
    arg_name: str,
) -> set[type]:
    """Get the type of the parameter for a function-like object.

    Return:
        A set containing the parameter's types. Empty set if the parameter
        does not exist or does not have a type hint.
    """
    globals_ = getattr(fn, "__globals__", {})
    locals_ = globals_

    # typing.get_type_hints does not work on 3.9 with A | None = None.
    # It is also overkill for retrieving type of a single parameter of
    # only function-like objects, and will cause problems
    # when other parameters have no type hint or type hint imported
    # conditioned on type_checking.

    param = inspect.signature(fn).parameters.get(arg_name, None)
    if param is None or param.annotation is inspect.Parameter.empty:
        return set()

    return _parse_types(param.annotation, globals_, locals_, is_py_39=_PY_V39)


def _get_default_argument(
    fn: typing.Callable,
    arg_name: str,
) -> typing.Any:  # noqa: ANN401
    """Get the default value of the parameter for a function-like object."""
    param = inspect.signature(fn).parameters.get(arg_name, None)
    if param is None:
        return inspect.Parameter.empty

    return param.default


def _convert_str_type(
    t: str,
    globals_,  # noqa: ANN001
    locals_,  # noqa: ANN001
) -> type:
    """Convert a type in a string form to a type object, handling Union types.

    !!! warning

        It does no handle types with type parameters in them.

    Raises:
        TypeConvertError if the type cannot be resolved.
    """
    args = [arg.strip() for arg in t.split("|")]
    try:
        type_args = [
            _eval_type(  # We pass type_params=None because https://github.com/python/cpython/issues/118418
                typing.ForwardRef(arg), globals_, locals_, type_params=None
            )
            for arg in args
        ]
    except (TypeError, SyntaxError, NameError) as e:
        raise TypeConvertError(f"Could not resolve type {t}.") from e
    return typing.Union[tuple(type_args)]


def _parse_types(
    type_hint: str | type,
    globals_: dict,
    locals_: dict,
    *,
    is_py_39: bool,
) -> set[type]:
    if isinstance(type_hint, str):
        opt_type = _convert_str_type(type_hint, globals_, locals_)
    else:
        opt_type = type_hint

    if _is_union_type(opt_type, is_py_39):
        return set(typing.get_args(opt_type))
    return {opt_type}


def _is_union_type(opt_type: type, is_py_39: bool) -> bool:  # noqa: FBT001
    return (
        is_py_39
        and typing.get_origin(opt_type) == typing.Union
        or (
            not is_py_39
            and typing.get_origin(opt_type) in (types.UnionType, typing.Union)
        )
    )


def _unwrap_wrapped_func(func: typing.Callable) -> typing.Callable:
    """Unwrap wrappers from a function."""
    while hasattr(func, "__wrapped__"):
        func = func.__wrapped__
    return func


T = typing.TypeVar("T")


def get_and_validate_param_type(
    fn: typing.Callable,
    type_check: type[T],
    parameter: str = "options",
) -> type[T] | None:
    """Get the type of the parameter for a function-like object.

    The function-like object must have an `parameter` with a type hint,
    following any of the following patterns:

        * `Union[type, None]`
        * `type | None`
        * `Optional[type]`.

    Returns:
        Type of the parameter if it exists and satisfies the above
        conditions, otherwise `None`.

    Raises:
        ValueError: When the type hint contains a subclass of `type_check`, but
            does not follow any of the specific patterns.
    """
    expected_args_length = 2
    opt_type = _get_argument_types(_unwrap_wrapped_func(fn), parameter)
    opt_default = _get_default_argument(_unwrap_wrapped_func(fn), parameter)
    compatible_types = [
        t for t in opt_type if isinstance(t, type) and issubclass(t, type_check)
    ]

    if compatible_types:
        if (
            len(opt_type) != expected_args_length
            or type(None) not in opt_type
            or opt_default is not None
        ):
            raise TypeError(
                "It seems like you want to use the workflow feature of automatically "
                "passing options to the tasks, but the type provided is wrong. "
                f"Please use either {compatible_types[0].__name__} | None = None, "
                f"Optional[{compatible_types[0].__name__}] = None or "
                f"Union[{compatible_types[0].__name__},None] = None "
                "to enable this feature.",
            )
        return compatible_types[0]
    return None


def check_type(
    value,  # noqa: ANN001
    t: type | str,
    globals_: dict,
    locals_: dict,
) -> bool:
    """Check if the value conforms to common types.

    The types are as follows:
    - Non-generic types: int, str, float, etc.
    - Union
    - Literal
    - Sequence: List, Tuple
    - Callable
    - Dict
    - Custom data classes

    For genetic types, only the origin is checked.


    Returns:
        True if the value conforms to the type hint or type cannot be resolved.
        False otherwise.
    """
    if isinstance(t, str):
        try:
            t = _convert_str_type(t, globals_, locals_)
        except TypeConvertError:
            # Type could not be resolved => skip the check
            return True

    if t in (object, typing.Any):
        return True
    origin = typing.get_origin(t)
    if origin is None:
        return isinstance(value, t)
    if origin is typing.Union:
        return any(isinstance(value, arg) for arg in typing.get_args(t))
    if origin is typing.Literal:
        return value in typing.get_args(t)
    # check containers  like List, Tuple, Set, Sequence
    # and Callable, Dict
    if origin in (list, tuple, set, typing.Sequence, typing.Callable, dict):
        return isinstance(value, origin)
    return True
