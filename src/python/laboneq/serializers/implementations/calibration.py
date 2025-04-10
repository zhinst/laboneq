# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.calibration.calibration import Calibration
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)

from ._models._calibration import (
    CalibrationModel,
    make_converter,
)

_converter = make_converter()


@serializer(types=Calibration, public=True)
class CalibrationSerializer(VersionedClassSerializer[Calibration]):
    SERIALIZER_ID = "laboneq.serializers.implementations.CalibrationSerializer"
    VERSION = 2

    @classmethod
    def to_dict(
        cls, obj: Calibration, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": _converter.unstructure(obj, CalibrationModel),
        }

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Calibration:
        return _converter.structure(serialized_data["__data__"], CalibrationModel)

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Calibration:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
