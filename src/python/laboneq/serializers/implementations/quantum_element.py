# Copyright 2024 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import attrs

from laboneq.dsl.quantum import QuantumElement, QuantumParameters
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.core import from_dict, import_cls, to_dict
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)


@serializer(types=[QuantumParameters], public=True)
class QuantumParametersSerializer(VersionedClassSerializer[QuantumParameters]):
    SERIALIZER_ID = "laboneq.serializers.implementations.QuantumParametersSerializer"
    VERSION = 2

    @classmethod
    def to_dict(
        cls, obj: QuantumParameters, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "quantum_parameters_class": f"{obj.__class__.__module__}.{obj.__class__.__name__}",
                "parameters": attrs.asdict(obj),
            },
        }

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> QuantumParameters:
        data = serialized_data["__data__"]
        qp_cls = import_cls(data["quantum_parameters_class"])
        return qp_cls(**from_dict(data["parameters"]))

    # v2 added the custom parameter to the QuantumParameters base class.
    # v1 is guaranteed to not have the custom parameter so that older versions
    # of LabOne Q that support v1 can load any v1 file.
    from_dict_v1 = from_dict_v2


class QuantumParametersContainer:
    """A class for identifying a list as a list of quantum parameters."""

    def __init__(self, quantum_parameters):
        self.quantum_parameters = quantum_parameters

    @classmethod
    def supports(cls, obj: object) -> bool:
        """Return true if `obj` is a tuple or list of QuantumParameters.

        Arguments:
            obj:
                The object to check.
        """
        if not isinstance(obj, (tuple, list)):
            return False
        return all(isinstance(x, QuantumParameters) for x in obj)


@serializer(types=QuantumParametersContainer, public=True)
class QuantumParametersContainerSerializer(
    VersionedClassSerializer[QuantumParametersContainer]
):
    SERIALIZER_ID = (
        "laboneq.serializers.implementations.QuantumParametersContainerSerializer"
    )
    VERSION = 1

    @classmethod
    def to_dict(
        cls,
        obj: QuantumParametersContainer,
        options: SerializationOptions | None = None,
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": [to_dict(q) for q in obj.quantum_parameters],
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> list[QuantumParameters]:
        data = serialized_data["__data__"]
        return [from_dict(q) for q in data]


class QuantumParametersDict:
    """A class for identifying a dict as a dict of quantum parameters."""

    def __init__(self, quantum_parameters: dict[str, QuantumParameters]):
        self.quantum_parameters = quantum_parameters

    @classmethod
    def supports(cls, obj: object) -> bool:
        """Return true if `obj` is a dict of QuantumParameters with string keys.

        Arguments:
            obj:
                The object to check.
        """
        if not isinstance(obj, dict):
            return False
        return all(
            isinstance(k, str) and isinstance(p, QuantumParameters)
            for k, p in obj.items()
        )


@serializer(types=QuantumParametersDict, public=True)
class QuantumParametersDictSerializer(VersionedClassSerializer[QuantumParametersDict]):
    SERIALIZER_ID = (
        "laboneq.serializers.implementations.QuantumParametersDictSerializer"
    )
    VERSION = 1

    @classmethod
    def to_dict(
        cls,
        obj: QuantumParametersDict,
        options: SerializationOptions | None = None,
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {k: to_dict(q) for k, q in obj.quantum_parameters.items()},
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> dict[str, QuantumParameters]:
        data = serialized_data["__data__"]
        return {k: from_dict(q) for k, q in data.items()}


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

    @classmethod
    def supports(cls, obj: object) -> bool:
        """Return true if `obj` is a tuple or list of QuantumElements.

        Arguments:
            obj:
                The object to check.
        """
        if not isinstance(obj, (tuple, list)):
            return False
        return all(isinstance(x, QuantumElement) for x in obj)


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


class QuantumElementDict:
    """A class for identifying a dict as a dict of quantum elements."""

    def __init__(self, quantum_elements: dict[str, QuantumElement]):
        self.quantum_elements = quantum_elements

    @classmethod
    def supports(cls, obj: object) -> bool:
        """Return true if `obj` is a dict of QuantumElements with string keys.

        Arguments:
            obj:
                The object to check.
        """
        if not isinstance(obj, dict):
            return False
        return all(
            isinstance(k, str) and isinstance(p, QuantumElement) for k, p in obj.items()
        )


@serializer(types=QuantumElementDict, public=True)
class QuantumElementDictSerializer(VersionedClassSerializer[QuantumElementDict]):
    SERIALIZER_ID = "laboneq.serializers.implementations.QuantumElementDictSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls,
        obj: QuantumElementDict,
        options: SerializationOptions | None = None,
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {k: to_dict(q) for k, q in obj.quantum_elements.items()},
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> dict[str, QuantumElement]:
        data = serialized_data["__data__"]
        return {k: from_dict(q) for k, q in data.items()}
