# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.quantum.qpu import QPU
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.core import import_cls, from_dict, to_dict
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    SerializationOptions,
    DeserializationOptions,
    JsonSerializableType,
)


@serializer(types=[QPU], public=True)
class QPUSerializer(VersionedClassSerializer[QPU]):
    SERIALIZER_ID = "laboneq.serializers.implementations.QPUSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: QPU, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        qop_cls = obj.quantum_operations.__class__
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "qubits": [to_dict(q) for q in obj.qubits],
                # We should use __qualname__ here but that complicates things
                # for import_cls
                "quantum_operations_class": f"{qop_cls.__module__}.{qop_cls.__name__}",
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> QPU:
        data = serialized_data["__data__"]
        qubits = [from_dict(q) for q in data["qubits"]]
        qop_cls = import_cls(data["quantum_operations_class"])
        qop = qop_cls()
        return QPU(
            qubits=qubits,
            quantum_operations=qop,
        )
