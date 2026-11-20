# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.dsl.parameter import LinearSweepParameter, SweepParameter
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer

from ._models._calibration import ParameterModel, make_converter

if TYPE_CHECKING:
    from laboneq.serializers.types import (
        DeserializationOptions,
        JsonSerializableType,
        SerializationOptions,
    )

_converter = make_converter()


@serializer(types=[SweepParameter, LinearSweepParameter], public=True)
class SweepParameterSerializer(
    VersionedClassSerializer[SweepParameter | LinearSweepParameter]
):
    SERIALIZER_ID = "laboneq.serializers.implementations.SweepParameterSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls,
        obj: SweepParameter | LinearSweepParameter,
        options: SerializationOptions | None = None,
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": _converter.unstructure(obj, ParameterModel),
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> SweepParameter | LinearSweepParameter:
        return _converter.structure(serialized_data["__data__"], ParameterModel)
