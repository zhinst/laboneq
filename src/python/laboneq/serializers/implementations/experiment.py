# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.core.types.enums.dsl_version import DSLVersion
from laboneq.dsl.experiment.experiment import Experiment
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)
from laboneq.serializers.implementations._models._experiment import (
    AllSectionModel,
    make_converter,
    ExperimentSignalModel,
)
from laboneq.serializers._pulse_cache import PulseSampledCache

_converter = make_converter()


@serializer(types=Experiment, public=True)
class ExperimentSerializer(VersionedClassSerializer[Experiment]):
    SERIALIZER_ID = "laboneq.serializers.implementations.ExperimentSerializer"
    VERSION = 2

    @classmethod
    def to_dict(
        cls, obj: Experiment, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        with PulseSampledCache.create_pulse_cache():
            sections = [
                _converter.unstructure(section, AllSectionModel)
                for section in obj.sections
            ]
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "uid": obj.uid,
                "name": obj.name,
                "signals": {
                    k: _converter.unstructure(v, ExperimentSignalModel)
                    for k, v in obj.signals.items()
                },
                "version": _converter.unstructure(obj.version),
                "epsilon": obj.epsilon,
                "sections": sections,
            },
        }

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Experiment:
        serialized_data = serialized_data["__data__"]
        with PulseSampledCache.create_pulse_cache():
            sections = [
                _converter.structure(section, AllSectionModel)
                for section in serialized_data["sections"]
            ]
        return Experiment(
            uid=serialized_data["uid"],
            name=serialized_data["name"],
            signals={
                k: _converter.structure(v, ExperimentSignalModel)
                for k, v in serialized_data["signals"].items()
            },
            version=_converter.structure(serialized_data["version"], DSLVersion),
            epsilon=serialized_data["epsilon"],
            sections=sections,
        )

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Experiment:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
