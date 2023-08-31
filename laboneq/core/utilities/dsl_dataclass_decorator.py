# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from io import StringIO

from rich.console import Console
from rich.pretty import pprint


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

    Examples:
        @classformatter
        @dataclasses.dataclass
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
    original_repr = cls.__repr__

    def __str__(self):
        with StringIO() as buffer:
            console = Console(file=buffer)
            pprint(self, console=console, expand_all=True, indent_guides=True)
            result = buffer.getvalue()
        return result

    def __format__(self, _):
        return original_repr(self)

    cls.__str__ = __str__
    cls.__format__ = __format__

    return cls
