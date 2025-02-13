# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Generic, TypeVar, cast


from laboneq._version import get_version
from laboneq.core.serialization.simple_serialization import (
    deserialize_from_dict_with_ref,
    serialize_to_dict_with_ref,
)
from laboneq.dsl.serialization.class_config import (
    classes_by_short_name,
    entity_config,
)
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    ClassSerializer,
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)


class UnwrappedData:
    """Container for unwrapped serialized data"""

    def __init__(self, data: JsonSerializableType):
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid serialization format. Expected a dict, got {type(data)}."
            )
        self.serializer = data["__serializer__"]
        self.version = data["__version__"]
        self.data: JsonSerializableType = data["__data__"]
        if not isinstance(self.serializer, str) or not isinstance(self.version, int):
            raise ValueError(
                "Invalid serialization format. Expected '__serializer__' and "
                "'__version__' to be str and int, got "
                f"'{serializer}'({type(serializer)}) and "
                f"'{self.version}'({type(self.version)})."
            )
        self.additional_data: dict[str, JsonSerializableType] = {
            k: v
            for k, v in data.items()
            if k not in ["__serializer__", "__version__", "__data__"]
        }


_T = TypeVar("_T")


class VersionedClassSerializer(ClassSerializer[_T]):
    """A serializer for a single type with multiple versions."""

    VERSION: int = 1
    SERIALIZER_ID: str = ""

    @classmethod
    def serializer_id(cls) -> str:
        return cls.SERIALIZER_ID

    @classmethod
    def version(cls) -> int:
        return cls.VERSION

    @classmethod
    def from_dict(
        cls,
        data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> _T:
        """Converts a dictionary to an object of the class."""
        unwrapped_data = UnwrappedData(data)
        from_dict = getattr(cls, f"from_dict_v{unwrapped_data.version}", None)
        if from_dict is None:
            raise ValueError(
                f"Version {unwrapped_data.version} is not supported by {cls.serializer_id()}."
            )
        return from_dict(data, options)


T = TypeVar("T")


class LabOneQClassicSerializer(Generic[T], VersionedClassSerializer[T]):
    """A serializer for the classic LabOne Q simple serialization."""

    @classmethod
    def to_dict(
        cls, obj: object, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        entity_classes, entity_mapper = entity_config()
        omit_none_fields = options.omit_none_fields if options else False
        serialized = serialize_to_dict_with_ref(
            obj,
            entity_classes,
            whitelist=classes_by_short_name(),
            entity_mapper=entity_mapper,
            emit_enum_types=True,
            omit_none_fields=omit_none_fields,
        ) | {"__version": get_version()}
        return cls._wrap(cast(JsonSerializableType, serialized))

    @classmethod
    def _wrap(
        serializer: type[ClassSerializer],
        data: JsonSerializableType,
        additional_data: dict[str, JsonSerializableType] | None = None,
    ) -> dict[str, JsonSerializableType]:
        return {
            "__serializer__": serializer.serializer_id(),
            "__version__": serializer.version(),
            "__data__": data,
        } | (additional_data or {})

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> T:
        """Converts a dictionary to an object of the class."""
        data = UnwrappedData(serialized_data)
        if data.additional_data and list(data.additional_data.keys()) != [
            "__creator__"
        ]:
            warnings.warn(
                f"Unexpected additional data during deserialization, keys: {list(data.additional_data.keys())}.",
                stacklevel=2,
            )
        if not isinstance(data.data, dict):
            raise ValueError(f"'data' is not a dictionary, got {type(data.data)}.")
        version = data.data.get("__version")
        if version is not None and version != get_version():
            warnings.warn(
                f"Deserializing data with version {version}, but current version is {get_version()}. This may lead to errors.",
                stacklevel=2,
            )

        entity_classes, entity_mapper = entity_config()

        return deserialize_from_dict_with_ref(
            data.data,
            classes_by_short_name(),
            entity_classes,
            entity_mapper,
        )
