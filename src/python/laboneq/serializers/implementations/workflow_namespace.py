# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""Serializer/Deserializer for workflow Namespace"""

from __future__ import annotations
from laboneq.workflow.blocks import Namespace
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq import serializers
from laboneq.serializers.types import (
    SerializationOptions,
    DeserializationOptions,
    JsonSerializableType,
)


@serializer(types=Namespace, public=True)
class WorkflowNamespaceSerializer(VersionedClassSerializer[Namespace]):
    SERIALIZER_ID = "laboneq.serializers.implementations.WorkflowNamespaceSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: Namespace, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {k: serializers.to_dict(v) for k, v in vars(obj).items()},
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Namespace:
        return Namespace(
            **{
                k: serializers.from_dict(v)
                for k, v in serialized_data["__data__"].items()
            }
        )
