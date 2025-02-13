# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Provides a factory for creating serializers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, TypeVar, overload


from laboneq.serializers.types import (
    ClassSerializer,
)


@dataclass
class TypeRecord:
    serializer_id: str
    public: bool


class SerializerRegistry:
    """A factory for creating serializers."""

    def __init__(self):
        self.serializers_by_id: dict[str, type[ClassSerializer[Any]]] = {}
        self.serializers_by_type: dict[type, TypeRecord] = {}

    def register(
        self,
        serializer_class: type[ClassSerializer[Any]],
        types: type | list[type] | None = None,
        public: bool = False,
    ):
        """Registers a serializer for a class.

        Arguments:
            serializer_class: The serializer to register
            types: Types which the serializer can convert to/from a dict
            public: Whether the types can saved on their own, i.e., at top level
        """
        if not isinstance(serializer_class, type):
            raise TypeError(
                f"Serializer of type {type(serializer_class)!r} is not a type/class."
            )
        if not isinstance(serializer_class, ClassSerializer):
            raise TypeError(
                f"Serializer of type {serializer_class!r} "
                "does not implement the ClassSerializer protocol."
            )
        serializer_id: str | None = serializer_class.serializer_id()
        if serializer_id is None:
            raise ValueError(f"Serializer {serializer_class} has no valid ID.")

        self.serializers_by_id[serializer_id] = serializer_class
        if types is not None:
            if isinstance(types, list):
                for t in types or []:
                    self.serializers_by_type[t] = TypeRecord(serializer_id, public)
            else:
                self.serializers_by_type[types] = TypeRecord(serializer_id, public)

    def __getitem__(self, serializer_id: object) -> type[ClassSerializer[Any]] | None:
        """Iterating through the MRO of the class, find the first registered serializer."""

        if isinstance(serializer_id, type):
            # Check if the given type derives any registered types.
            for obj_type in serializer_id.mro():
                if record := self.serializers_by_type.get(obj_type, None):
                    return self[record.serializer_id]
            return None
        if isinstance(serializer_id, str):
            return self.serializers_by_id.get(serializer_id, None)
        raise ValueError(f"Invalid serializer ID/type {serializer_id!r}")

    def is_public(self, type: type) -> bool:
        """Returns whether the type can be serialized at the top level.
        Iterating through the MRO of the class, find the first registered serializer and
        return whether it is public.
        If no serializer is found, return False."""

        for obj_type in type.mro():
            if record := self.serializers_by_type.get(obj_type, None):
                return record.public
        return False


# The global factory
serializer_registry = SerializerRegistry()

T = TypeVar("T")


@overload
def serializer(
    cls: type[ClassSerializer[T]],
    *,
    types: type | list[type] | None = ...,
    public: bool = ...,
    registry: SerializerRegistry = ...,
) -> type[ClassSerializer[T]]: ...


@overload
def serializer(
    cls: None = ...,
    *,
    types: type | list[type] | None = ...,
    public: bool = ...,
    registry: SerializerRegistry = ...,
) -> Callable[[type[ClassSerializer[T]]], type[ClassSerializer[T]]]: ...


def serializer(
    cls: type[ClassSerializer[T]] | None = None,
    *,
    types: type | list[type] | None = None,
    public: bool = False,
    registry: SerializerRegistry = serializer_registry,
) -> (
    type[ClassSerializer[T]]
    | Callable[[type[ClassSerializer[T]]], type[ClassSerializer[T]]]
):
    """Decorator for registering a Serializer.

    Arguments:
        types: The types that the serializer can handle.
        registry: The serializer registry to use.
    """

    if cls is None:

        def _register(cls: type[ClassSerializer[T]]) -> type[ClassSerializer[T]]:
            registry.register(cls, types=types, public=public)
            return cls

        return _register
    if not isinstance(cls, ClassSerializer):
        raise TypeError(
            f"Class {cls!r} is not a serializer. Use the 'types' keyword to specify types handled by this serializer."
        )
    registry.register(cls, types=types, public=public)
    return cls
