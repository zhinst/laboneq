# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import contextlib
import inspect
from collections.abc import Callable
from typing import Union, get_origin, get_args
from types import UnionType

from multimethod import multimethod, subtype, DispatchError, signature

from laboneq.dsl.quantum.quantum_element import QuantumElement


def _type_signature_length(func: Callable) -> int:
    """Return the length of a function type signature."""
    sig = inspect.signature(func)
    return len(sig.parameters)


def _simplify_type_signature(types: tuple) -> tuple:
    """Replace non-QuantumElement types with object in the type signature.

    For union types, simplify each member of the union:
    - If all members are QuantumElement subclasses, preserve the union
    - If some members are QuantumElement subclasses and others are not,
      replace non-QuantumElement members with object
    - If no members are QuantumElement subclasses, replace the entire union with object
    """
    simplified_types = []
    for t in types:
        if inspect.isclass(t) and issubclass(t, QuantumElement):
            simplified_types.append(t)
        elif get_origin(t) is UnionType:  # Handle union types
            union_members = get_args(t)
            simplified_members = []

            for member in union_members:
                if inspect.isclass(member) and issubclass(member, QuantumElement):
                    simplified_members.append(member)
                else:
                    simplified_members.append(object)

            simplified_types.append(Union[tuple(simplified_members)])
        else:
            simplified_types.append(object)

    return tuple(simplified_types)


def _is_type_compatible(call_type: type, expected_type: type) -> bool:
    """Check if call_type is compatible with expected_type."""
    # Standard subclass check
    try:
        if issubclass(call_type, expected_type):
            return True
    except TypeError:
        # expected_type might not be a normal class
        pass

    # Handle union types (e.g. Transmon | AlternativeTransmon)
    if get_origin(expected_type) is UnionType:
        return any(
            _is_type_compatible(call_type, union_member)
            for union_member in get_args(expected_type)
        )

    return False


class MultiMethod(multimethod):
    """A custom subclass of multimethod for the LabOne Q DSL.

    The original `multimethod` package can be found [here](https://pypi.org/project/multimethod/).

    Changes include:
    - No auto-registering signatures during dispatch.
    - `copy` method copies attributes, as well as methods.
    - Non-`QuantumElement` types are converted to `object`.
    - Can only register type signatures of equal length.

    Methods overwritten (in order of appearance):
    - `register`
    - `copy`
    - `__setitem__`
    - `__missing__`
    - `__repr__`
    """

    def register(self, func: Callable, *types) -> Callable:
        """Register a function with type signature equal length validation."""
        if len(self) > 0:  # Only validate if there are existing signatures
            new_length = _type_signature_length(func)
            existing_lengths = set()

            for existing_func in self.values():
                existing_lengths.add(_type_signature_length(existing_func))

            if existing_lengths and new_length not in existing_lengths:
                existing_length = next(iter(existing_lengths))
                raise ValueError(
                    f"All type signatures in a MultiMethod must have the same length. "
                    f"Invalid type signature length: {new_length}. "
                    f"Expected type signature length: {existing_length}."
                )

        return super().register(func, *types)

    def copy(self):
        """Return a new multimethod with the same methods and attributes."""
        new_multimethod = super().copy()
        new_multimethod.__name__ = self.__name__
        new_multimethod.pending = self.pending.copy()
        new_multimethod.generics = self.generics.copy()
        new_multimethod._quantum_op = self._quantum_op.copy()
        return new_multimethod

    def __setitem__(self, types: tuple, func: Callable):
        # (This code is adapted from multimethod.multimethod.__setitem__)
        self.clean()
        types = _simplify_type_signature(types)  # <--- line added here
        if not isinstance(types, signature):
            types = signature(types)
        parents = types.parents = self.parents(types)
        with contextlib.suppress(ValueError):
            types.sig = inspect.signature(func)
        self.pop(types, None)  # ensure key is overwritten
        for key in self:
            if types < key and (not parents or parents & key.parents):
                key.parents -= parents
                key.parents.add(types)
        for index, cls in enumerate(types):
            if origins := set(subtype.origins(cls)):
                self.generics += [()] * (index + 1 - len(self.generics))
                self.generics[index] = tuple(origins.union(self.generics[index]))
        super().__setitem__(types, func)
        self.__doc__ = self.docstring

    def __missing__(self, types: tuple) -> Callable:
        """Find and return the next applicable method without caching the types."""
        # Check if the type signature is already in the multimethod cache
        # (This code is adapted from multimethod.multimethod.__missing__)
        self.evaluate()
        types = _simplify_type_signature(types)  # <--- line added here
        types = tuple(map(subtype, types))
        if types in self:
            return self[types]

        # Check if the type signature is matched using standard inheritance
        # (This code is adapted from multimethod.multimethod.__missing__)
        parents = self.parents(types)
        if parents:  # <--- line added here
            return self.select(types, parents)

        # Check for other compatible type signatures
        compatible = {
            sig
            for sig in self.keys()
            if isinstance(sig, tuple)
            and len(types) <= len(sig)
            and all(
                _is_type_compatible(call_type, sig[i])
                for i, call_type in enumerate(types)
            )
        }
        if compatible:
            return self.select(types, compatible)

        raise DispatchError(
            f"{getattr(self, '__name__', 'multimethod')}: 0 methods found", types, set()
        )

    def __repr__(self) -> str:
        mylist = []
        for key, value in self.items():
            key = tuple([k.__name__ for k in key])
            mylist.append(f"{key}: {value.__qualname__}")
        return f"<{type(self).__qualname__}\n" + "\n".join(mylist) + "\n>"
