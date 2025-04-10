# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import warnings

from laboneq._version import get_version
from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.core import from_dict, to_dict
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)
from laboneq.serializers.implementations._models._compiled_experiment import (
    make_converter,
    ScheduledExperimentModel,
)

_converter = make_converter()


@serializer(types=CompiledExperiment, public=True)
class CompiledExperimentSerializer(VersionedClassSerializer[CompiledExperiment]):
    SERIALIZER_ID = "laboneq.serializers.implementations.CompiledExperimentSerializer"
    VERSION = 2

    @classmethod
    def to_dict(
        cls, obj: CompiledExperiment, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        device_setup = to_dict(obj.device_setup, options)
        experiment = to_dict(obj.experiment, options)
        experiment_dict = to_dict(obj.experiment_dict, options)
        scheduled_experiment = _converter.unstructure(
            obj.scheduled_experiment, ScheduledExperimentModel
        )
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__laboneq_version__": get_version(),
            "__data__": {
                "device_setup": device_setup,
                "experiment": experiment,
                "experiment_dict": experiment_dict,
                "scheduled_experiment": scheduled_experiment,
            },
        }

    @classmethod
    def _check_laboneq_version(
        cls,
        serialized_laboneq_version: str | None,
        options: DeserializationOptions | None = None,
    ) -> None:
        check_version = options is None or not options.force
        _not_found = "Could not find LabOne Q version in serialized data."
        _mismatch = (
            f"LabOne Q version mismatch. Check out the Labone Q with correct version "
            f"{serialized_laboneq_version} to load the serialized data. Otherwise, set "
            f"the `force` option to True to skip the version check."
        )
        if serialized_laboneq_version is None:
            if check_version:
                raise ValueError(_not_found)
            else:
                warnings.warn(_not_found, UserWarning, stacklevel=2)
        elif serialized_laboneq_version != get_version():
            if check_version:
                raise ValueError(_mismatch)
            else:
                warnings.warn(_mismatch, UserWarning, stacklevel=2)

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> CompiledExperiment:
        cls._check_laboneq_version(serialized_data.get("__laboneq_version__"), options)

        device_setup = from_dict(serialized_data["__data__"]["device_setup"], options)
        experiment = from_dict(serialized_data["__data__"]["experiment"], options)
        experiment_dict = from_dict(
            serialized_data["__data__"]["experiment_dict"], options
        )
        scheduled_experiment = _converter.structure(
            serialized_data["__data__"]["scheduled_experiment"],
            ScheduledExperimentModel,
        )
        return CompiledExperiment(
            device_setup=device_setup,
            experiment=experiment,
            experiment_dict=experiment_dict,
            scheduled_experiment=scheduled_experiment,
        )

    @classmethod
    def from_dict_v1(
        cls, serialized_data, options: DeserializationOptions | None = None
    ) -> CompiledExperiment:
        cls._check_laboneq_version(serialized_data.get("__laboneq_version__"), options)
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
