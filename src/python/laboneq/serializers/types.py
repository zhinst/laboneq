# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations


from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, Union


from numpy import integer
from typing_extensions import TypeAlias, runtime_checkable


if TYPE_CHECKING:
    from laboneq.serializers.serializer_registry import SerializerRegistry

# This type represents what orjson can serialize to JSON
JsonBasicTypes: TypeAlias = Union[str, int, float, bool, integer[Any], None]
JsonSerializableType: TypeAlias = Union[
    JsonBasicTypes, dict[str, "JsonSerializableType"], list["JsonSerializableType"]
]


@dataclass
class SerializationOptions:
    """Options for serializing (storing) objects.

    Attributes:
        serializer_registry: The registry from which to look up the serializer
        omit_none_fields: Do not store fields which are None; only supported by
            some serializers
    """

    serializer_registry: SerializerRegistry | None = None
    omit_none_fields: bool = False


@dataclass
class DeserializationOptions:
    """Options for deserializing (loading) objects."""

    serializer_registry: SerializerRegistry | None = None


T = TypeVar("T")


@runtime_checkable
class ClassSerializer(Protocol, Generic[T]):
    """A serializer for a single type."""

    @classmethod
    def serializer_id(cls) -> str:
        """The ID of the serializer. Do not change after its first release."""
        ...

    @classmethod
    def version(cls) -> int:
        """The current version of the serializer."""
        ...

    @classmethod
    def to_json_dict(
        cls, obj: T, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        """Converts an object to JSON serializable format."""
        raise NotImplementedError(
            "Class serializers must implement .to_json_dict() if .to_dict() returns "
            "JSON unserializable object."
        )

    @classmethod
    def to_dict(
        cls, obj: T, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        """Converts the object to a dictionary."""
        raise NotImplementedError("Class serializers must implement .to_dict()")

    @classmethod
    def from_dict(
        cls, data: JsonSerializableType, options: DeserializationOptions | None = None
    ) -> T:
        """Converts a dictionary to an object of the class."""
        raise NotImplementedError("Class serializers must implement .from_dict()")
