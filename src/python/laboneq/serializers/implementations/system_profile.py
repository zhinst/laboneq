# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from laboneq.dsl.device import SystemProfile
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.implementations._models._device_setup import (
    SystemProfileModel,
    make_converter,
)
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)

_converter = make_converter()


@serializer(types=SystemProfile, public=True)
class SystemProfileSerializer(VersionedClassSerializer[SystemProfile]):
    SERIALIZER_ID = "laboneq.serializers.implementations.SystemProfileSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: SystemProfile, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        system_profile = _converter.unstructure(obj, SystemProfileModel)
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "system_profile": system_profile,
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> SystemProfile:
        assert isinstance(serialized_data, dict)
        sys_prof_data: dict[str, Any] = serialized_data["__data__"].get(
            "system_profile"
        )
        assert isinstance(sys_prof_data, dict)
        return _converter.structure(sys_prof_data, SystemProfileModel)
