# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from laboneq.dsl.device import SystemDescription
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.implementations._models._device_setup import (
    SystemDescriptionModel,
    make_converter,
)
from laboneq.serializers.serializer_registry import serializer

if TYPE_CHECKING:
    from laboneq.serializers.types import (
        DeserializationOptions,
        JsonSerializableType,
        SerializationOptions,
    )

_converter = make_converter()


@serializer(types=SystemDescription, public=True)
class SystemDescriptionSerializer(VersionedClassSerializer[SystemDescription]):
    SERIALIZER_ID = "laboneq.serializers.implementations.SystemDescriptionSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: SystemDescription, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        system_description = _converter.unstructure(obj, SystemDescriptionModel)
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "system_description": system_description,
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> SystemDescription:
        assert isinstance(serialized_data, dict)
        sys_prof_data: dict[str, Any] = serialized_data["__data__"].get(
            "system_description"
        )
        assert isinstance(sys_prof_data, dict)
        return _converter.structure(sys_prof_data, SystemDescriptionModel)
