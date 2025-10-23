# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from io import StringIO

from rich.console import Console
from rich.pretty import pprint


def _classformatter(cls):
    """The internals of the `classformatter` decorator.

    !!! note
        By itself, this decorator does not automatically apply to `attrs` subclasses
        which are explicitly decorated with `attrs.define`.
    """

    def __str__(self):
        f = StringIO()
        # Here we force jupyter detection to False so that what is printed is
        # not *also* sent to Jupyter in addition to the specified file.
        console = Console(file=f, force_jupyter=False)
        pprint(self, console=console, expand_all=True, indent_guides=True)
        return f.getvalue()

    def __format__(self, _):
        return repr(self)

    def repr_pretty(self, p, _cycle):  # noqa: ANN001, ANN202
        # For Notebooks
        p.text(str(self))

    cls.__str__ = __str__
    cls.__format__ = __format__
    cls._repr_pretty_ = repr_pretty  # pylint: disable=protected-access

    return cls


def classformatter(cls):
    """
    A decorator to customize the string representation of class instances using the rich library.

    This decorator overwrites the __str__ and __format__ methods of the decorated class. The new
    __str__ method pretty-prints the instance using rich's pprint function, ensuring a visually
    appealing output on compatible terminals. The __format__ method is overridden to return the
    class's original __repr__ representation.

    Args:
        cls (type): The class to be decorated.

    Returns:
        type: The decorated class with modified __str__ and __format__ methods.

    !!! version-changed "Changed in version 2.61.0"
        Changed the decorator, such that it automatically applies to `attrs` subclasses
        which are explicitly decorated with `attrs.define`.

    Examples:
        @classformatter
        @attrs.define
        class Person:
            name: str
            age: int
            friends: List["Person"]

        p = Person("John", 42, [Person("Alice", 23, []), Person("Bob", 25, [])])
        print(p)  # This will now print using the custom __str__ method.

        the result is:

        Person(
        │   name='John',
        │   age=42,
        │   friends=[
        │   │   Person(
        │   │   │   name='Alice',
        │   │   │   age=23,
        │   │   │   friends=[]
        │   │   ),
        │   │   Person(
        │   │   │   name='Bob',
        │   │   │   age=25,
        │   │   │   friends=[]
        │   │   )
        │   ]
        )
    """

    @classmethod
    def attrs_init_subclass(subcls):
        _classformatter(subcls)

    cls.__attrs_init_subclass__ = attrs_init_subclass
    _classformatter(cls)

    return cls
