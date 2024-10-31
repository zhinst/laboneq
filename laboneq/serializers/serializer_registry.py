# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Provides a factory for creating serializers."""

from __future__ import annotations

from typing import Any, Callable, TypeVar, overload

from laboneq.serializers.types import ClassSerializer


class SerializerRegistry:
    """A factory for creating serializers."""

    def __init__(self):
        self.serializers_by_id: dict[str, type[ClassSerializer[Any]]] = {}
        self.serializers_by_type: dict[type, str] = {}

    def register(
        self,
        serializer_class: type[ClassSerializer[Any]],
        types: type | list[type] | None = None,
    ):
        """Registers a serializer for a class."""
        serializer_id: str | None = serializer_class.serializer_id()
        if serializer_id is None:
            raise ValueError(f"Serializer {serializer_class} has no valid ID.")

        self.serializers_by_id[serializer_id] = serializer_class
        if types is not None:
            if isinstance(types, list):
                for t in types or []:
                    self.serializers_by_type[t] = serializer_id
            else:
                self.serializers_by_type[types] = serializer_id

    def __getitem__(self, serializer_id: object) -> type[ClassSerializer[Any]] | None:
        """Returns the serializer for the class."""

        if isinstance(serializer_id, type):
            return self[self.serializers_by_type.get(serializer_id, "")]
        if isinstance(serializer_id, str):
            return self.serializers_by_id.get(serializer_id, None)
        raise ValueError(f"Invalid serializer ID/type {serializer_id!r}")


# The global factory
serializer_registry = SerializerRegistry()

T = TypeVar("T")


@overload
def serializer(
    cls: type[ClassSerializer[T]],
    *,
    types: type | list[type] | None = ...,
    registry: SerializerRegistry = ...,
) -> type[ClassSerializer[T]]: ...


@overload
def serializer(
    cls: None = ...,
    *,
    types: type | list[type] | None = ...,
    registry: SerializerRegistry = ...,
) -> Callable[[type[ClassSerializer[T]]], type[ClassSerializer[T]]]: ...


# Decorator for registering a Serializer
def serializer(
    cls: type[ClassSerializer[T]] | None = None,
    *,
    types: type | list[type] | None = None,
    registry: SerializerRegistry = serializer_registry,
) -> (
    type[ClassSerializer[T]]
    | Callable[[type[ClassSerializer[T]]], type[ClassSerializer[T]]]
):
    """Decorator for registering a Serializer

    Arguments:
        types: The types that the serializer can handle.
        registry: The serializer registry to use.
    """

    if cls is None:

        def _register(cls: type[ClassSerializer[T]]) -> type[ClassSerializer[T]]:
            registry.register(cls, types=types)
            return cls

        return _register
    if not isinstance(cls, ClassSerializer):
        raise TypeError(
            f"Class {cls!r} is not a serializer. Use the 'types' keyword to specify types handled by this serializer."
        )
    registry.register(cls, types)
    return cls
