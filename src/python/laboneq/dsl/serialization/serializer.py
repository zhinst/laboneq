# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, TypeVar

import orjson

from laboneq.core.exceptions import LabOneQException
from laboneq.core.serialization.simple_serialization import (
    deserialize_from_dict_with_ref,
    serialize_to_dict_with_ref,
)
from laboneq.dsl.serialization.class_config import (
    classes_by_short_name,
    classes_by_short_name_ir,
    entity_config,
)


T = TypeVar("T")


class Serializer:
    @staticmethod
    def to_json_struct(serializable_object, omit_none_fields=False) -> str:
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
    def to_json(serializable_object, omit_none_fields=False) -> str:
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
    def to_dict(serializable_object, omit_none_fields=False) -> Dict:
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
    def to_json_file(serializable_object, filename: str):
        json_string = Serializer.to_json(serializable_object)
        try:
            with open(filename, mode="w") as file:
                file.write(json_string)
        except IOError as e:
            raise LabOneQException() from e

    @staticmethod
    def from_json(serialized_string: str, type_hint):
        if type_hint is dict:
            obj = orjson.loads(serialized_string)
        else:
            entity_classes, entity_mapper = entity_config()
            serialized_form = orjson.loads(serialized_string)

            obj = deserialize_from_dict_with_ref(
                serialized_form,
                classes_by_short_name_ir()
                if type_hint is not None and type_hint.__name__ == "IR"
                else classes_by_short_name(),
                entity_classes,
                entity_mapper,
            )

        return obj

    @staticmethod
    def load(data, type_hint: type[T]) -> T:
        if type_hint is dict:
            obj = copy.deepcopy(data)
        else:
            entity_classes, entity_mapper = entity_config()

            obj = deserialize_from_dict_with_ref(
                data, classes_by_short_name(), entity_classes, entity_mapper
            )

        return obj

    @staticmethod
    def from_json_file(filename: str, type_hint: type[T]) -> T:
        try:
            with open(filename, mode="r") as file:
                json_string = file.read()
        except IOError as e:
            raise LabOneQException(e.__repr__()) from e

        return Serializer.from_json(json_string, type_hint)
