# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, TypeVar

import orjson
from typing_extensions import deprecated

from laboneq.core.exceptions import LabOneQException
from laboneq.serializers._legacy.class_config import (
    classes_by_short_name,
    entity_config,
)
from laboneq.serializers._legacy.simple_serialization import (
    deserialize_from_dict_with_ref,
    serialize_to_dict_with_ref,
)

T = TypeVar("T")


def _uses_new_serializer(obj_or_type) -> bool:
    """Whether `obj_or_type` is a type the classic tree-walk no longer handles.

    `CompiledExperiment` holds the compiler's serialized output, which only
    `laboneq.serializers` knows how to carry. Everything else still tree-walks.
    """
    from laboneq.core.types.compiled_experiment import CompiledExperiment

    if isinstance(obj_or_type, type):
        return issubclass(obj_or_type, CompiledExperiment)
    return isinstance(obj_or_type, CompiledExperiment)


def _new_to_dict(obj, omit_none_fields: bool) -> Dict:
    from laboneq.serializers import to_dict
    from laboneq.serializers.types import SerializationOptions

    return to_dict(obj, SerializationOptions(omit_none_fields=omit_none_fields))


def _new_from_dict(data):
    from laboneq.serializers import from_dict

    return from_dict(data)


class Serializer:
    """
    Legacy LabOne Q serializer.

    !!! version-changed "Deprecated in version 26.4"
        This serializer will be removed in a future release.
        Use `laboneq.serializers` instead.
    """

    @staticmethod
    @deprecated(
        "The laboneq.dsl.serialization.Serializer class is deprecated. Use `laboneq.serializers.to_json` instead.",
        category=FutureWarning,
    )
    def to_json_struct(serializable_object, omit_none_fields=False) -> dict:
        if _uses_new_serializer(serializable_object):
            return _new_to_dict(serializable_object, omit_none_fields)

        entity_classes, entity_mapper = entity_config()

        json_struct = serialize_to_dict_with_ref(
            serializable_object,
            entity_classes,
            whitelist=classes_by_short_name(),
            entity_mapper=entity_mapper,
            emit_enum_types=True,
            omit_none_fields=omit_none_fields,
        )

        return json_struct

    @staticmethod
    @deprecated(
        "The laboneq.dsl.serialization.Serializer class is deprecated. Use `laboneq.serializers.to_json` instead.",
        category=FutureWarning,
    )
    def to_json(serializable_object, omit_none_fields=False) -> str:
        if _uses_new_serializer(serializable_object):
            from laboneq.serializers import to_json

            return to_json(serializable_object).decode()

        options = orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY
        try:
            if isinstance(serializable_object, dict):
                json_dump = orjson.dumps(serializable_object, option=options)
            else:
                entity_classes, entity_mapper = entity_config()

                json_struct = serialize_to_dict_with_ref(
                    serializable_object,
                    entity_classes,
                    whitelist=classes_by_short_name(),
                    entity_mapper=entity_mapper,
                    emit_enum_types=True,
                    omit_none_fields=omit_none_fields,
                )
                json_dump = orjson.dumps(json_struct, option=options)
            return json_dump.decode()
        except TypeError as ex:
            raise LabOneQException("Cannot serialize object to json") from ex

    @staticmethod
    @deprecated(
        "The laboneq.dsl.serialization.Serializer class is deprecated. Use `laboneq.serializers.to_dict` instead.",
        category=FutureWarning,
    )
    def to_dict(serializable_object, omit_none_fields=False) -> Dict:
        if _uses_new_serializer(serializable_object):
            return _new_to_dict(serializable_object, omit_none_fields)

        entity_classes, entity_mapper = entity_config()

        return serialize_to_dict_with_ref(
            serializable_object,
            entity_classes,
            whitelist=classes_by_short_name(),
            entity_mapper=entity_mapper,
            emit_enum_types=True,
            omit_none_fields=omit_none_fields,
        )

    @staticmethod
    @deprecated(
        "The laboneq.dsl.serialization.Serializer class is deprecated. Use `laboneq.serializers.from_json` instead.",
        category=FutureWarning,
    )
    def from_json(serialized_string: str, type_hint):
        if _uses_new_serializer(type_hint):
            from laboneq.serializers import from_json

            return from_json(serialized_string)

        if type_hint is dict:
            obj = orjson.loads(serialized_string)
        else:
            entity_classes, entity_mapper = entity_config()
            serialized_form = orjson.loads(serialized_string)

            obj = deserialize_from_dict_with_ref(
                serialized_form,
                classes_by_short_name(),
                entity_classes,
                entity_mapper,
            )

        return obj

    @staticmethod
    @deprecated(
        "The laboneq.dsl.serialization.Serializer class is deprecated. Use `laboneq.serializers.load` instead.",
        category=FutureWarning,
    )
    def load(data, type_hint: type[T]) -> T:
        if _uses_new_serializer(type_hint):
            return _new_from_dict(data)

        if type_hint is dict:
            obj = copy.deepcopy(data)
        else:
            entity_classes, entity_mapper = entity_config()

            obj = deserialize_from_dict_with_ref(
                data, classes_by_short_name(), entity_classes, entity_mapper
            )

        return obj
