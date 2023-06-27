# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from rich.console import Console
from rich.pretty import pprint


def classformatter(cls):
    """
    Decorator that sets the _repr_html_ and __str__ methods to custom functions.
    """

    def __str__(self):
        """
        Get a plain string representation of an object, with custom formatting for class names.
        Args:
            self: The object to get a pretty string of.
        Returns:
            A string containing the pretty string of the object, with custom class name formatting.
        """
        console = Console(record=True, quiet=False)
        with console.capture() as capture:
            pprint(self, console=console, expand_all=True, indent_guides=True)
        return capture.get()

    cls.__str__ = __str__

    return cls
