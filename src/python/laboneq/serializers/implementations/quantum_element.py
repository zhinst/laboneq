# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs

from laboneq.dsl.quantum import QuantumElement
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.core import import_cls, from_dict, to_dict
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    SerializationOptions,
    DeserializationOptions,
    JsonSerializableType,
)


@serializer(types=[QuantumElement], public=True)
class QuantumElementSerializer(VersionedClassSerializer[QuantumElement]):
    SERIALIZER_ID = "laboneq.serializers.implementations.QuantumElementSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: QuantumElement, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                # We should use __qualname__ here but that complicates things
                # for import_cls
                "quantum_element_class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "uid": obj.uid,
                "signals": obj.signals,
                # Base quantum element does not have parameters but this will change with the new qubit
                # class.
                "parameter_class": f"{obj.parameters.__class__.__module__}.{obj.parameters.__class__.__name__}",
                "parameters": attrs.asdict(obj.parameters),
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> QuantumElement:
        data = serialized_data["__data__"]
        qe_cls = import_cls(data["quantum_element_class"])
        param_cls = import_cls(data["parameter_class"])
        return qe_cls(
            uid=data["uid"],
            signals=data["signals"],
            parameters=param_cls(**from_dict(data["parameters"])),
        )


class QuantumElementContainer:
    """A class for identifying a list as a list of quantum elements."""

    def __init__(self, quantum_elements):
        self.quantum_elements = quantum_elements


@serializer(types=QuantumElementContainer, public=True)
class QuantumElementContainerSerializer(VersionedClassSerializer[QuantumElement]):
    SERIALIZER_ID = (
        "laboneq.serializers.implementations.QuantumElementContainerSerializer"
    )
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: QuantumElementContainer, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": [to_dict(q) for q in obj.quantum_elements],
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> list[QuantumElement]:
        data = serialized_data["__data__"]
        return [from_dict(q) for q in data]
