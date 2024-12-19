# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Serializer/Deserializer for laboneq.workflow.WorkflowOptions"""

from __future__ import annotations

import attrs

from laboneq import serializers
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.core import import_cls
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)
from laboneq.workflow import TaskOptions, WorkflowOptions


@serializer(types=WorkflowOptions, public=True)
class WorkflowOptionsSerializer(VersionedClassSerializer[WorkflowOptions]):
    SERIALIZER_ID = "laboneq.serializers.implementations.WorkflowOptionsSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: WorkflowOptions, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        # We let attrs handle core Python object types as they are not currently supported
        # by the LabOne Q serializer (e.g a dict).
        def attr_filter(attr: attrs.Attribute, _):
            return False if attr.name == "logstore" else True

        def attr_serializer(_, attr: attrs.Attribute, value: object):
            if attr and attr.name == "_task_options":
                return {k: serializers.to_dict(v) for k, v in value.items()}
            try:
                return serializers.to_dict(value)
            except RuntimeError:
                return value

        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                **attrs.asdict(
                    obj, value_serializer=attr_serializer, filter=attr_filter
                ),
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> WorkflowOptions:
        data = serialized_data["__data__"].copy()
        return import_cls(data.pop("class__"))(
            _task_options={
                k: serializers.from_dict(v)
                for k, v in data.pop("_task_options").items()
            },
            **{k: serializers.from_dict(v) for k, v in data.items()},
        )


@serializer(types=TaskOptions, public=True)
class TaskOptionsSerializer(VersionedClassSerializer[TaskOptions]):
    SERIALIZER_ID = "laboneq.serializers.implementations.TaskOptionsSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: TaskOptions, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        # We let attrs handle core Python object types as they are not currently supported
        # by the LabOne Q serializer (e.g a dict).
        def attr_serializer(_, attr: attrs.Attribute, value: object):
            try:
                return serializers.to_dict(value)
            except RuntimeError:
                return value

        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "class__": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                **attrs.asdict(obj, value_serializer=attr_serializer),
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> TaskOptions:
        data = serialized_data["__data__"].copy()
        return import_cls(data.pop("class__"))(
            **{k: serializers.from_dict(v) for k, v in data.items()},
        )
