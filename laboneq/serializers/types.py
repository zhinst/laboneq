# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, Protocol, Union

from numpy import integer
from typing_extensions import TypeAlias, runtime_checkable

# This type represents what orjson can serialize to JSON
JsonBasicTypes: TypeAlias = Union[str, int, float, bool, integer[Any], None]
JsonSerializableType: TypeAlias = Union[
    JsonBasicTypes, dict[str, "JsonSerializableType"], list["JsonSerializableType"]
]


@runtime_checkable
class ClassSerializer(Protocol):
    """A serializer for a single type."""

    @classmethod
    def serializer_id(cls) -> str:
        """The ID of the serializer. Do not change after its first release."""
        ...

    @classmethod
    def version(cls) -> int:
        """The current version of the serializer."""
        ...
