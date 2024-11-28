# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Base options for workflow."""

from __future__ import annotations

from functools import wraps
import sys
import typing
from io import StringIO
from typing import Callable, Type, final

import attr
from rich.console import Console
from rich.pretty import pprint

from laboneq.workflow.opts.options_parser import check_type
from attrs import setters


def _type_validator(inst, attr: attr.Attribute, value) -> None:  # noqa: ANN001
    """A validator for common type checking.

    Raises:
        TypeError:
            With a human readable error message, including the attribute name
            the expected type, and the value it got.
    """
    module_name = inst.__class__.__module__

    module = sys.modules[module_name]

    globals_ = module.__dict__
    locals_ = globals_
    if not check_type(value, attr.type, globals_, locals_):
        msg = (
            f"'{attr.name}' must be {attr.type!r} (got {value!r} that is of "
            f"type {type(value)!r})."
        )
        raise TypeError(msg)


def option_field(
    default: typing.Any = attr.NOTHING,  # noqa: ANN401
    *,
    factory: typing.Callable[[], typing.Any] | None = None,
    validators: list[typing.Callable[[typing.Any, attr.Attribute, typing.Any], None]]
    | None = None,
    description: str | None = None,
    exclude: bool = False,
    eq: bool = False,
    alias: str | None = None,
    converter: typing.Callable[[typing.Any], typing.Any] | None = None,
    repr: bool = True,  # noqa: A002
) -> attr.Attribute:
    """Create a field for an options class.

    Attributes:
        default:
            The default value of the field.
        factory:
            The factory to use for the field.
        validators:
            The validators to use for the field. When provided, only the
            custom validators are used. When not provided or None,
            a basic validator is used that performs type checking for
            the following types:
            - Non-generic types: int, str, float, etc.
            - Union, Optional
            - Generic types: List, Dict, Tuple, Set, Callable, etc. Only
            the origin is checked.
            - User-defined classes

        description:
            The description of the field.
        exclude:
            Whether to exclude the field from serialization.
        eq:
            Whether to include the field in the equality check.
        alias:
            The alias of the field.
        converter:
            The converter to use for the field.
        repr:
            Whether to include the field in the representation.
    """
    if validators is None:
        validators = [_type_validator]
    return attr.field(
        default=default,
        factory=factory,
        validator=validators,
        alias=alias,
        eq=eq,
        repr=repr,
        converter=converter,
        metadata={"description": description, "exclude": exclude},
    )


class BaseOptions:
    """Base class for all Option classes."""

    def __str__(self):
        with StringIO() as buffer:
            console = Console(file=buffer)
            pprint(self, console=console, expand_all=True, indent_guides=True)
            return buffer.getvalue()

    @final
    def __format__(self, _):  # noqa: ANN001
        return self.__repr__()

    @final
    def _repr_pretty_(self, p, _cycle):  # noqa: ANN001, ANN202
        # For Notebooks
        p.text(str(self))

    @final
    def __repr__(self):
        return self.__str__()

    def to_dict(self) -> dict:
        """Generate a dictionary representation of the options."""
        return attr.asdict(self, filter=self._exclude_fields_to_dict)

    @classmethod
    def from_dict(cls, data: dict) -> BaseOptions:
        """Create an instance of the class from a dictionary."""
        return cls(**data)

    def _exclude_fields_to_dict(self, attribute, _) -> bool:  # noqa: ANN001
        return not attribute.metadata.get("exclude", False)

    @property
    def fields(self) -> dict:
        """Return public fields of the options class."""
        d = attr.fields_dict(self.__class__)
        # remove the private fields
        return {k: v for k, v in d.items() if not k.startswith("_")}


def options(
    cls: Type | None = None,
    *,
    base_class: Type[BaseOptions] = BaseOptions,
) -> type:
    """Decorator to register a class as an options class.

    Args:
        cls:
            The class to decorate or None if using keyword arguments.
        base_class:
            The base_class for the decorated class.
            Default is `BaseOptions`.
            Must be a subclass of `BaseOptions`.
    """
    DEFAULT_SETATTR = setters.pipe(setters.convert, setters.validate)

    @wraps(cls)
    def wrapper(cls):
        cls = attr.define(cls, repr=False, auto_attribs=True, slots=False)
        if not issubclass(cls, base_class):
            _module = cls.__module__
            # by default, attr.make_class does not call validators and converters when fields are set.
            cls = attr.make_class(
                cls.__name__, {}, bases=(cls, base_class), on_setattr=DEFAULT_SETATTR
            )
            # attr.make_class will change the module of the class, so we need to set it back for serialization
            # to work correctly.
            cls.__module__ = _module

        if not all(field.default is not attr.NOTHING for field in attr.fields(cls)):
            raise ValueError(f"All fields in {cls.__name__} must have default values.")
        return cls

    return _options_decorator(cls, base_class, BaseOptions, wrapper)


def _options_decorator(
    cls: Type | None, base_class: Type, default_base_class: Type, wrapper: Callable
) -> type:
    if not issubclass(base_class, default_base_class):
        raise TypeError(
            f"Base class must be a subclass of {default_base_class.__name__}."
        )

    def decorator(cls: Type) -> Type:
        return wrapper(cls)

    if cls is not None:
        return decorator(cls)

    return decorator
