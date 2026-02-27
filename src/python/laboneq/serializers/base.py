# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TypeVar

from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    ClassSerializer,
    DeserializationOptions,
    JsonSerializableType,
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
