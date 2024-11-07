# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""A decorator for adding pretty print to class objects."""

from io import StringIO
from typing import Any, Protocol, TypeVar

use_rich_pprint: bool = True
try:
    from rich.console import Console
    from rich.pretty import pprint as rich_pprint
except ImportError:
    from pprint import pprint as pprint_pprint

    use_rich_pprint = False


class HasAsStrDict(Protocol):
    """A protocol for classes that have a method to return a dictionary representation.

    The dictionary representation is used to pretty-print the object using the pprint
    function. The dictionary should contain all attributes that should be printed.
    Not to be confused with the __dict__ attribute, which contains the actual attributes
    of the object, including methods and private attributes.
    """

    def _as_str_dict(self) -> dict[str, Any]: ...


T = TypeVar("T", bound=HasAsStrDict)


def classformatter(cls: type[T]) -> type[T]:
    """A decorator to customize the string representation of class instances.

    This decorator overwrites the __str__ and __format__ methods of the decorated class.
    The new __str__ method pretty-prints the instance using the pprint function,
    ensuring a visually appealing output on compatible terminals. The __format__ method
    is overridden to return the class's original __repr__ representation. Also,
    the `_repr_pretty_` method is added to support pretty-printing in Jupyter notebooks.

    In contrast to the similar decorator with the same name from laboneq, this creates
    the string representation based on a dictionary provided by the class'
    (   )`_as_str_dict()`.

    If the global variable `use_rich_pprint` is `True`, the rich library will be used,
    otherwise `pprint.pprint`.

    Args:
        cls (type): The class to be decorated.

    Returns:
        type: The decorated class with modified __str__ and __format__ methods.
    """

    def new_str(self: T) -> str:
        as_dict = self._as_str_dict()
        with StringIO() as buffer:
            if use_rich_pprint:
                console = Console(file=buffer)
                rich_pprint(
                    as_dict,
                    console=console,
                    expand_all=True,
                    indent_guides=True,
                )
            else:
                pprint_pprint(as_dict, stream=buffer)  # noqa: T203
            return buffer.getvalue()

    def new_format(self: T, _: object) -> str:
        return format(self._as_str_dict())

    def repr_pretty(self, p, _cycle):  # noqa: ANN001, ANN202
        # For Notebooks
        p.text(str(self))

    cls.__str__ = new_str
    cls.__format__ = new_format
    cls._repr_pretty_ = repr_pretty

    return cls
