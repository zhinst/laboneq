# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

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
    remove_high_pass_clearing,
)

_logger = logging.getLogger(__name__)
_converter = make_converter()


@serializer(types=Calibration, public=True)
class CalibrationSerializer(VersionedClassSerializer[Calibration]):
    SERIALIZER_ID = "laboneq.serializers.implementations.CalibrationSerializer"
    VERSION = 3

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
    def from_dict_v3(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Calibration:
        serialized_data = serialized_data["__data__"]
        return _converter.structure(serialized_data, CalibrationModel)

    @classmethod
    def _remove_high_pass_clearing_v2(cls, calibration_items: dict):
        for signal_uid, signal_info in calibration_items.items():
            remove_high_pass_clearing(signal_uid, signal_info, _logger)

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Calibration:
        se = serialized_data["__data__"]
        cls._remove_high_pass_clearing_v2(se["calibration_items"])
        return cls.from_dict_v3(serialized_data, options)

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Calibration:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
