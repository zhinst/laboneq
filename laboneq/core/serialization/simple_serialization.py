# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib
import inspect
import logging
from collections.abc import Iterable, Mapping
from enum import Enum
from io import BytesIO
from typing import Dict

import numpy as np
import pybase64 as base64
from numpy.lib.format import read_array, write_array
from sortedcontainers import SortedDict

ID_KEY = "__id"


class NumpyArrayRepr:
    def __new__(
        cls,
        array_data=None,
        real_data=None,
        complex_data=None,
        binary_npz=None,
        binary_npy=None,
    ):
        # deserialize
        if binary_npz is not None:
            # For backwards compatibility only, we no longer emit npz blobs
            input_buffer = BytesIO(base64.b64decode(binary_npz.encode("ascii")))
            loaded_npz = np.load(input_buffer)
            return loaded_npz[loaded_npz.files[0]]
        if binary_npy is not None:
            input_buffer = base64.b64decode(binary_npy.encode("ascii"))
            return read_array(BytesIO(input_buffer))
        if real_data is not None:
            return np.array(real_data)
        if complex_data is not None:
            return np.array(complex_data).astype(complex)

        # serialize array_data
        return super().__new__(cls)

    def __init__(
        self,
        array_data=None,
        # Must specify all arguments, as keys are filtered based on function signature
        real_data=None,
        complex_data=None,
        binary_npz=None,
        binary_npy=None,
    ):
        assert array_data is not None
        assert real_data is complex_data is binary_npz is binary_npy is None
        if array_data.size > 100:
            output_buffer = BytesIO()
            write_array(output_buffer, array_data, version=(3, 0), allow_pickle=False)
            output_buffer.seek(0)

            self.binary_npy = base64.b64encode(output_buffer.read()).decode("ascii")
        else:
            if np.isrealobj(array_data):
                self.real_data = array_data.tolist()
            else:
                self.complex_data = np.frompyfunc(str, 1, 1)(array_data).tolist()


def full_classname(klass):
    module = klass.__module__
    if module == "builtins":
        return klass.__qualname__
    return module + "." + klass.__qualname__


def full_typename(o):
    klass = o.__class__
    return full_classname(klass)


def short_typename(o):
    return type(o).__name__


def module_classes(modules_list):
    classes_by_fullname = SortedDict()
    classes_by_short_name = SortedDict()
    for x in modules_list:
        module = importlib.import_module(x)
        for name, member in inspect.getmembers(module):
            if inspect.isclass(member):
                classes_by_fullname[full_classname(member)] = member
                classes_by_short_name[member.__name__] = member
    return classes_by_fullname, classes_by_short_name


def serialize_to_dict(to_serialize, emit_enum_types=False):

    if to_serialize is None:
        return
    if (
        isinstance(to_serialize, str)
        or isinstance(to_serialize, float)
        or isinstance(to_serialize, int)
    ):
        return to_serialize

    if isinstance(to_serialize, Enum):
        if emit_enum_types:
            retval = {
                "__type": full_typename(to_serialize),
                "enum.value": to_serialize.value,
            }
            return retval
        else:
            return str(to_serialize)

    if isinstance(to_serialize, np.ndarray):
        to_serialize = to_serialize.tolist()

    if isinstance(to_serialize, list):
        retval = [serialize_to_dict(item, emit_enum_types) for item in to_serialize]
        return retval

    mapping = {}
    sub_dict = {}
    dir_list = _dir(to_serialize.__class__)
    is_object = False
    if hasattr(to_serialize, "__dict__"):
        is_object = True
        mapping = to_serialize.__dict__
        sub_dict["__type"] = full_typename(to_serialize)
    elif isinstance(to_serialize, Mapping):
        mapping = to_serialize

    for k, v in mapping.items():
        outkey = k
        outvalue = v
        if k[0] == "_" and is_object:
            if k[1:] in dir_list:
                outkey = k[1:]
                outvalue = getattr(to_serialize, outkey)
            else:
                continue
        if isinstance(outvalue, Mapping):
            sub_dict[outkey] = serialize_to_dict(v, emit_enum_types)
        elif isinstance(outvalue, Iterable) and not isinstance(outvalue, str):
            sub_dict[outkey] = []
            index = 0
            for item in outvalue:
                sub_dict[outkey].append(serialize_to_dict(item, emit_enum_types))
                index += 1
        else:
            sub_dict[outkey] = serialize_to_dict(outvalue, emit_enum_types)
    sub_dict = {k: v for k, v in sub_dict.items() if v is not None}
    return sub_dict


@functools.lru_cache()
def class_argnames(cls):
    return inspect.signature(cls.__init__).parameters.keys()


def construct_object(content, mapped_class):
    if len(content.keys()) == 1 and _issubclass(mapped_class, Enum):
        return mapped_class(list(content.values())[0])
    arg_names = class_argnames(mapped_class)
    has_kwargs = "kwargs" in arg_names
    filtered_args = {}
    for k, v in content.items():

        if k in arg_names:
            filtered_args[k] = v
        elif has_kwargs and isinstance(v, Mapping):
            filtered_args.update(v)
        else:
            logging.getLogger(__name__).debug(
                f"Ignoring field {k} in {mapped_class} because it is not in the __init__ method"
            )
    return mapped_class(**filtered_args)


def create_ref(item, item_ref_type):
    return {"$ref": item.uid, "__entity_type": item_ref_type}


def serialize_to_dict_with_entities(
    to_serialize, entity_classes, entities_collector, emit_enum_types=False
):
    cls = to_serialize.__class__
    if to_serialize is None:
        return None
    if _issubclass(cls, (bool, int, float, str)):
        return to_serialize

    if _issubclass(cls, Enum):
        if emit_enum_types:
            return {
                "__type": short_typename(to_serialize),
                "enum.value": to_serialize.value,
            }
        else:
            return str(to_serialize)

    if _issubclass(cls, np.ndarray):
        return serialize_to_dict_with_entities(
            NumpyArrayRepr(array_data=to_serialize),
            entity_classes,
            entities_collector,
            emit_enum_types,
        )

    if _issubclass(cls, list):
        retval = []
        for item in to_serialize:
            item_is_entity, item_entity_class = is_entity_class(
                item.__class__, entity_classes
            )
            if item_is_entity:
                entity_typename_full = full_classname(item_entity_class)
                entity_typename_short = entity_typename_full.split(".")[-1]
                if item.uid is None:
                    raise RuntimeError(
                        f"Entities must have valid uid, but {item} has uid of None"
                    )
                if entity_typename_short not in entities_collector:
                    entities_collector[entity_typename_short] = {}

                if item.uid not in entities_collector[entity_typename_short]:
                    entities_collector[entity_typename_short][
                        item.uid
                    ] = serialize_to_dict_with_entities(
                        item, entity_classes, entities_collector, emit_enum_types
                    )
                    entities_collector[entity_typename_short][item.uid][ID_KEY] = id(
                        item
                    )
                retval.append(create_ref(item, entity_typename_short))
            else:
                retval.append(
                    serialize_to_dict_with_entities(
                        item, entity_classes, entities_collector, emit_enum_types
                    )
                )
        return retval

    mapping = {}
    sub_dict = {}
    dir_list = _dir(to_serialize.__class__)
    is_object = False
    if _issubclass(cls, Mapping):
        mapping = to_serialize
    elif hasattr(to_serialize, "__dict__"):
        is_object = True
        item_is_entity, item_entity_class = is_entity_class(
            to_serialize.__class__, entity_classes
        )
        if item_is_entity:
            if to_serialize.uid is None:
                raise RuntimeError(f"to_serialize has uid of None, {to_serialize}")

        mapping = to_serialize.__dict__
        sub_dict["__type"] = short_typename(to_serialize)

    for k, v in mapping.items():
        item_class = v.__class__

        outkey, outvalue = k, v
        if is_object and k[0] == "_":
            if k[1:] in dir_list:
                outkey = k[1:]
                outvalue = getattr(to_serialize, outkey)
            else:
                continue
        if item_class in (bool, int, float, str) or v is None:
            sub_dict[outkey] = outvalue
            continue
        item_is_entity, item_entity_class = is_entity_class(item_class, entity_classes)
        if item_is_entity:
            entity_typename_full = full_classname(item_entity_class)
            entity_typename_short = entity_typename_full.split(".")[-1]
            if v.uid is None:
                raise RuntimeError(
                    f"Entities must have valid uid, but {v} has uid of None"
                )
            if entity_typename_short not in entities_collector:
                entities_collector[entity_typename_short] = {}
            if v.uid not in entities_collector[entity_typename_short]:
                entities_collector[entity_typename_short][
                    v.uid
                ] = serialize_to_dict_with_entities(
                    v, entity_classes, entities_collector, emit_enum_types
                )
                entities_collector[entity_typename_short][v.uid][ID_KEY] = id(v)
            else:
                if entities_collector[entity_typename_short][v.uid][ID_KEY] != id(v):
                    raise RuntimeError(
                        f"uid not unique: item {v} has same uid as previously encountered item {entities_collector[entity_typename_short][v.uid]}"
                    )

            sub_dict[outkey] = create_ref(v, entity_typename_short)
        elif _issubclass(item_class, np.ndarray):
            sub_dict[outkey] = serialize_to_dict_with_entities(
                NumpyArrayRepr(array_data=outvalue),
                entity_classes,
                entities_collector,
                emit_enum_types,
            )

        elif _issubclass(item_class, Mapping):
            sub_dict[outkey] = serialize_to_dict_with_entities(
                v, entity_classes, entities_collector, emit_enum_types
            )
        elif _issubclass(item_class, Iterable) and not _issubclass(item_class, str):
            sub_dict[outkey] = []
            for item in outvalue:
                item_is_entity, item_entity_class = is_entity_class(
                    item.__class__, entity_classes
                )
                if item_is_entity:
                    if item.uid is None:
                        raise RuntimeError(
                            f"Entities must have valid uid, but {v} has uid of None"
                        )
                    entity_typename_full = full_classname(item_entity_class)
                    entity_typename_short = entity_typename_full.split(".")[-1]

                    if entity_typename_short not in entities_collector:
                        entities_collector[entity_typename_short] = {}
                    if item.uid not in entities_collector[entity_typename_short]:
                        entities_collector[entity_typename_short][
                            item.uid
                        ] = serialize_to_dict_with_entities(
                            item,
                            entity_classes,
                            entities_collector,
                            emit_enum_types,
                        )
                        entities_collector[entity_typename_short][item.uid][
                            ID_KEY
                        ] = id(item)

                    else:
                        if entities_collector[entity_typename_short][item.uid][
                            ID_KEY
                        ] != id(item):
                            raise RuntimeError(
                                f"uid not unique: item {item} has same uid as previously encountered item {entities_collector[entity_typename_short][item.uid]}"
                            )
                    sub_dict[outkey].append(create_ref(item, entity_typename_short))
                else:
                    sub_dict[outkey].append(
                        serialize_to_dict_with_entities(
                            item,
                            entity_classes,
                            entities_collector,
                            emit_enum_types,
                        )
                    )
        else:
            sub_dict[outkey] = serialize_to_dict_with_entities(
                outvalue, entity_classes, entities_collector, emit_enum_types
            )
    if is_object:
        sub_dict = {k: v for k, v in sub_dict.items() if v is not None}
    return sub_dict


def serialize_to_dict_with_ref(
    to_serialize, entity_classes, entity_mapper=None, emit_enum_types=False
):
    if entity_mapper is None:
        entity_mapper = {}
    entities_collector = {}
    root_object = serialize_to_dict_with_entities(
        to_serialize,
        entity_classes,
        entities_collector,
        emit_enum_types=emit_enum_types,
    )

    for k, v in entities_collector.items():
        for uid, entity in v.items():
            if ID_KEY in entity:
                # remove memory id from serialization - only required during serialization to identify non-unique uids
                del entity[ID_KEY]

    entities_collector_flat = {
        entity_mapper[k] if k in entity_mapper else k: list(v.values())
        for k, v in entities_collector.items()
    }
    root_name = full_typename(to_serialize).split(".")[-1].lower()
    retval = {root_name: root_object}
    if entities_collector_flat:
        retval["entities"] = entities_collector_flat

    return retval


@functools.lru_cache()
def is_entity_class(candidate_class, entity_class_list):
    for entity_class in entity_class_list:
        if issubclass(candidate_class, entity_class):
            return True, entity_class
    return False, None


@functools.lru_cache()
def _issubclass(cls, base):
    return issubclass(cls, base)


@functools.lru_cache()
def _dir(cls):
    return dir(cls)


def deserialize_from_dict_with_ref_recursor(
    data, class_mapping, entity_pool_raw: Dict, entity_pool_deserialized: Dict
):
    cls = data.__class__
    if cls is dict:
        if ("$ref" in data) and ("__entity_type" in data):
            key = data["$ref"], data["__entity_type"]
            return entity_pool_deserialized.setdefault(
                key,
                deserialize_from_dict_with_ref_recursor(
                    entity_pool_raw[key],
                    class_mapping,
                    entity_pool_raw,
                    entity_pool_deserialized,
                ),
            )
        out_mapping = {
            k: deserialize_from_dict_with_ref_recursor(
                v, class_mapping, entity_pool_raw, entity_pool_deserialized
            )
            if v.__class__ not in (bool, int, float, str)
            else v
            # for performance: do not recurse on simple primitives
            for k, v in data.items()
            if k != "__type"
        }
        type_name = data.get("__type")
        if type_name is None:
            return out_mapping
        type_name_short = type_name.split(".")[-1]
        mapped_class = class_mapping.get(type_name_short)
        if mapped_class is None:
            raise Exception(
                f"Class {type_name_short} / {type_name} not found, known classes are {list(class_mapping.keys())}"
            )
        constructed_object = construct_object(out_mapping, mapped_class)
        return constructed_object
    if cls is list:
        return [
            deserialize_from_dict_with_ref_recursor(
                item, class_mapping, entity_pool_raw, entity_pool_deserialized
            )
            if item.__class__ not in (bool, int, float, str)
            else item
            # for performance: do not recurse on simple primitives
            for item in data
        ]
    return data


def deserialize_from_dict_with_ref(data, class_mapping, entity_classes, entity_map):
    class_mapping[NumpyArrayRepr.__name__] = NumpyArrayRepr
    entity_pool = {}

    for _, entity_list in data.get("entities", {}).items():
        for entity in entity_list:
            type_name = entity["__type"]
            type_name_short = type_name.split(".")[-1]
            mapped_class = class_mapping.get(type_name_short)
            current_class_is_entity_class, entity_class = is_entity_class(
                mapped_class, entity_classes
            )
            if not current_class_is_entity_class:
                continue
            entity_class_name_short = full_classname(entity_class).split(".")[-1]
            entity_key = entity["uid"], entity_class_name_short
            if entity_key in entity_pool:
                raise RuntimeError(
                    f"Non-unique UID {entity['uid']} for type {entity['__type']}"
                )
            entity_pool[entity_key] = entity

    root_key = next(k for k, v in data.items() if k != "entities")
    entity_pool[("__ROOT__", "")] = data[root_key]
    entity_pool_deserialized = {}
    return deserialize_from_dict_with_ref_recursor(
        data[root_key], class_mapping, entity_pool, entity_pool_deserialized
    )


def deserialize_from_dict(data, class_mapping):
    cls = data.__class__
    if _issubclass(cls, Mapping):
        out_mapping = {}
        for k, v in data.items():
            if k != "__type":
                out_mapping[k] = deserialize_from_dict(v, class_mapping)
        type_name = data.get("__type")
        if type_name is not None:
            type_name_short = type_name.split(".")[-1]
            mapped_class = class_mapping.get(type_name_short)
            if mapped_class is None:
                raise Exception(
                    f"Class {type_name_short} / {type_name} not found, known classes are {list(class_mapping.keys())}"
                )
            return construct_object(out_mapping, mapped_class)
        else:
            return out_mapping
    elif _issubclass(cls, Iterable) and not _issubclass(cls, str):
        out_list = []
        for item in data:
            out_list.append(deserialize_from_dict(item, class_mapping))
        return out_list
    else:
        return data
