# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import Generic, TypeVar, cast

from laboneq._version import get_version
from laboneq.serializers._legacy.class_config import (
    classes_by_short_name,
    entity_config,
)
from laboneq.serializers._legacy.simple_serialization import (
    deserialize_from_dict_with_ref,
    serialize_to_dict_with_ref,
)
from laboneq.serializers.base import UnwrappedData, VersionedClassSerializer
from laboneq.serializers.types import (
    ClassSerializer,
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)

T = TypeVar("T")


class LabOneQClassicSerializer(Generic[T], VersionedClassSerializer[T]):
    """A serializer for the classic LabOne Q simple serialization."""

    @classmethod
    def to_dict(
        cls, obj: object, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        entity_classes, entity_mapper = entity_config()
        omit_none_fields = options.omit_none_fields if options else False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
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
        cls: type[ClassSerializer],
        data: JsonSerializableType,
        additional_data: dict[str, JsonSerializableType] | None = None,
    ) -> dict[str, JsonSerializableType]:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            deserialized = deserialize_from_dict_with_ref(
                data.data,
                classes_by_short_name(),
                entity_classes,
                entity_mapper,
            )

        return deserialized
