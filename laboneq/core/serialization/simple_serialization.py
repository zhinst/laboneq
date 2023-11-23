# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib
import inspect
import logging
from collections.abc import Mapping
from enum import Enum
from io import BytesIO, StringIO
from typing import Dict

import numpy as np
import pybase64 as base64
from numpy.lib.format import read_array, write_array
from sortedcontainers import SortedDict

from laboneq.core.serialization.externals import (
    XarrayDataArrayDeserializer,
    XarrayDatasetDeserializer,
    serialize_maybe_xarray,
)

_logger = logging.getLogger(__name__)

ID_KEY = "__id"


class SerializerException(Exception):
    pass


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


def module_classes(modules_list, class_names=None):
    classes_by_fullname = SortedDict()
    classes_by_short_name = SortedDict()
    for x in modules_list:
        module = importlib.import_module(x)
        for _name, member in inspect.getmembers(module):
            if inspect.isclass(member):
                if class_names is None or member.__name__ in class_names:
                    classes_by_fullname[full_classname(member)] = member
                    classes_by_short_name[member.__name__] = member
    return classes_by_fullname, classes_by_short_name


@functools.lru_cache()
def all_slots(cls):
    slots = set()
    for base in cls.__mro__:
        if isinstance(getattr(base, "__slots__", []), str):
            slots.add(getattr(base, "__slots__", []))
        else:
            for attr in getattr(base, "__slots__", []):
                slots.add(attr)
    return list(slots)


__fields_cache = dict()


def all_fields(obj):
    if short_typename(obj) == "NumpyArrayRepr":
        return (
            [fname for fname in obj.__dict__.keys()] if hasattr(obj, "__dict__") else []
        )
    if obj.__class__ not in __fields_cache:
        __fields_cache[obj.__class__] = (
            [fname for fname in obj.__dict__.keys()] if hasattr(obj, "__dict__") else []
        )
    return __fields_cache[obj.__class__]


def get_fields_and_slots(obj):
    slots = all_slots(type(obj))
    fields = all_fields(obj)
    return slots + fields


def serialize_to_dict(
    to_serialize, emit_enum_types=False, omit_none_fields=False, depth=0, max_depth=100
):
    if depth == max_depth:
        raise SerializerException(
            f"Reached maximum recursion depth of {max_depth} in serialization"
        )
    depth += 1
    if to_serialize is None:
        return
    if isinstance(to_serialize, (str, float, int, np.int64, np.complex128, complex)):
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
        return NumpyArrayRepr(array_data=to_serialize)

    if isinstance(to_serialize, list):
        return [
            serialize_to_dict(item, emit_enum_types, depth=depth, max_depth=max_depth)
            for item in to_serialize
        ]

    if isinstance(to_serialize, (tuple, set)):
        retval = {
            "__type": full_typename(to_serialize),
            "__content": [
                serialize_to_dict(
                    item, emit_enum_types, depth=depth, max_depth=max_depth
                )
                for item in to_serialize
            ],
        }
        if isinstance(to_serialize, set):
            retval["__content"] = sorted(retval["__content"])
        return retval

    mapping = {}
    sub_dict = {}
    dir_list = _dir(to_serialize.__class__)
    is_object = False

    if hasattr(to_serialize, "__dict__") or hasattr(to_serialize, "__slots__"):
        is_object = True
        all_attrs_slots = get_fields_and_slots(to_serialize)
        mapping = {a_or_s: getattr(to_serialize, a_or_s) for a_or_s in all_attrs_slots}
        sub_dict["__type"] = full_typename(to_serialize)

    elif isinstance(to_serialize, Mapping):
        mapping = to_serialize

    for k, v in mapping.items():
        outkey = k
        outvalue = v
        if is_object and k[0] == "_":
            if k[1:] in dir_list:
                outkey = k[1:]
                outvalue = getattr(to_serialize, outkey)
            else:
                continue
        if isinstance(outvalue, Mapping):
            sub_dict[outkey] = serialize_to_dict(
                v, emit_enum_types, depth=depth, max_depth=max_depth
            )
        elif isinstance(outvalue, list):
            sub_dict[outkey] = []
            for item in outvalue:
                sub_dict[outkey].append(
                    serialize_to_dict(
                        item, emit_enum_types, depth=depth, max_depth=max_depth
                    )
                )
        else:
            sub_dict[outkey] = serialize_to_dict(
                outvalue, emit_enum_types, depth=depth, max_depth=max_depth
            )

    if is_object and omit_none_fields:
        sub_dict = {k: v for k, v in sub_dict.items() if v is not None}

    return sub_dict


@functools.lru_cache()
def class_argnames(cls):
    return inspect.signature(cls.__init__).parameters.keys()


def construct_object(content, mapped_class):
    if len(content.keys()) == 1 and _issubclass(mapped_class, Enum):
        return mapped_class(list(content.values())[0])
    if _issubclass(mapped_class, XarrayDataArrayDeserializer):
        return mapped_class(content)
    if _issubclass(mapped_class, XarrayDatasetDeserializer):
        return mapped_class(content)
    arg_names = class_argnames(mapped_class)
    has_kwargs = "kwargs" in arg_names
    filtered_args = {}
    for k, v in content.items():
        if k in arg_names:
            filtered_args[k] = v
        elif has_kwargs and isinstance(v, Mapping):
            filtered_args.update(v)
        else:
            _logger.debug(
                f"Ignoring field {k} in {mapped_class} because it is not in the __init__ method"
            )
    return mapped_class(**filtered_args)


def create_ref(item, item_ref_type):
    return {"$ref": item.uid, "__entity_type": item_ref_type}


def serialize_to_dict_with_entities(
    to_serialize,
    entity_classes,
    whitelist,
    entities_collector,
    emit_enum_types=False,
    omit_none_fields=False,
    depth=0,
    max_depth=100,
):
    if depth == max_depth:
        raise SerializerException(
            f"Reached maximum recursion depth of {max_depth} in serialization"
        )
    depth += 1
    cls = to_serialize.__class__
    if to_serialize is None:
        return None
    if _issubclass(cls, (bool, int, float, str, np.int64)):
        return to_serialize

    if _issubclass(cls, (np.complex128, complex)):
        return {
            "__type": "np.complex128",
            "real": to_serialize.real,
            "imag": to_serialize.imag,
        }

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
            whitelist,
            entities_collector,
            emit_enum_types,
            omit_none_fields,
            depth,
            max_depth,
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
                        item,
                        entity_classes,
                        whitelist,
                        entities_collector,
                        emit_enum_types,
                        omit_none_fields,
                        depth,
                        max_depth,
                    )
                    entities_collector[entity_typename_short][item.uid][ID_KEY] = id(
                        item
                    )
                retval.append(create_ref(item, entity_typename_short))
            else:
                retval.append(
                    serialize_to_dict_with_entities(
                        item,
                        entity_classes,
                        whitelist,
                        entities_collector,
                        emit_enum_types,
                        omit_none_fields,
                        depth,
                        max_depth,
                    )
                )
        return retval

    if _issubclass(cls, (set, tuple)):
        sub_dict = {}
        sub_dict["__type"] = full_typename(to_serialize)
        sub_dict["__contents"] = []
        for cotent in to_serialize:
            sub_dict["__contents"].append(
                serialize_to_dict_with_entities(
                    cotent,
                    entity_classes,
                    whitelist,
                    entities_collector,
                    emit_enum_types,
                    omit_none_fields,
                    depth,
                    max_depth,
                )
            )
        if _issubclass(cls, set):
            sub_dict["__contents"] = sorted(sub_dict["__contents"])
        return sub_dict

    # Optional dependency `xarray` object serialization
    if (
        xarr := serialize_maybe_xarray(
            obj=to_serialize,
            serializer_function=serialize_to_dict_with_entities,
            entity_classes=entity_classes,
            entities_collector=entities_collector,
            emit_enum_types=emit_enum_types,
            omit_none_fields=omit_none_fields,
        )
    ) is not None:
        return xarr

    mapping = {}
    sub_dict = {}
    dir_list = _dir(to_serialize.__class__)
    is_object = False

    if _issubclass(cls, Mapping):
        mapping = to_serialize

    elif hasattr(to_serialize, "__dict__") or hasattr(to_serialize, "__slots__"):
        is_object = True
        item_is_entity, item_entity_class = is_entity_class(
            to_serialize.__class__, entity_classes
        )
        if item_is_entity:
            if to_serialize.uid is None:
                raise RuntimeError(f"to_serialize has uid of None, {to_serialize}")

        all_attrs_slots = get_fields_and_slots(to_serialize)
        mapping = {a_or_s: getattr(to_serialize, a_or_s) for a_or_s in all_attrs_slots}
        type_name = short_typename(to_serialize)

        if type_name not in whitelist:
            _logger.warning(f"instance of class {type_name} may not serialize properly")

        sub_dict["__type"] = type_name

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
                    v,
                    entity_classes,
                    whitelist,
                    entities_collector,
                    emit_enum_types,
                    omit_none_fields,
                    depth,
                    max_depth,
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
                whitelist,
                entities_collector,
                emit_enum_types,
                omit_none_fields,
                depth,
                max_depth,
            )

        elif _issubclass(item_class, Mapping):
            sub_dict[outkey] = serialize_to_dict_with_entities(
                v,
                entity_classes,
                whitelist,
                entities_collector,
                emit_enum_types,
                omit_none_fields,
                depth,
                max_depth,
            )
        elif _issubclass(item_class, list):
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
                            whitelist,
                            entities_collector,
                            emit_enum_types,
                            omit_none_fields,
                            depth,
                            max_depth,
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
                            whitelist,
                            entities_collector,
                            emit_enum_types,
                            omit_none_fields,
                            depth,
                            max_depth,
                        )
                    )
        else:
            sub_dict[outkey] = serialize_to_dict_with_entities(
                outvalue,
                entity_classes,
                whitelist,
                entities_collector,
                emit_enum_types,
                omit_none_fields,
                depth,
                max_depth,
            )

    if is_object and omit_none_fields:
        sub_dict = {k: v for k, v in sub_dict.items() if v is not None}

    return sub_dict


def serialize_to_dict_with_ref(
    to_serialize,
    entity_classes,
    whitelist,
    entity_mapper=None,
    emit_enum_types=False,
    omit_none_fields=False,
    max_depth=100,
):
    if entity_mapper is None:
        entity_mapper = {}
    entities_collector = {}

    log_stream = StringIO()
    log_handler = logging.StreamHandler(log_stream)
    logging.getLogger("laboneq.core.serialization.simple_serialization").addHandler(
        log_handler
    )
    try:
        root_object = serialize_to_dict_with_entities(
            to_serialize,
            entity_classes,
            whitelist,
            entities_collector,
            emit_enum_types=emit_enum_types,
            omit_none_fields=omit_none_fields,
            depth=0,
            max_depth=max_depth,
        )
    except Exception as ex:
        unique_log_msgs = "\n".join({l for l in log_stream.getvalue().splitlines()})
        if len(unique_log_msgs) > 0:
            raise SerializerException(
                f"The following warning(s) were encountered during serialization:\n{unique_log_msgs}"
            ) from ex
        else:
            raise ex

    for v in entities_collector.values():
        for entity in v.values():
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
        if type_name_short == "set":
            return {
                deserialize_from_dict_with_ref_recursor(
                    item, class_mapping, entity_pool_raw, entity_pool_deserialized
                )
                if item.__class__ not in (bool, int, float, str)
                else item
                # for performance: do not recurse on simple primitives
                for item in data["__contents"]
            }
        if type_name_short == "tuple":
            return tuple(
                deserialize_from_dict_with_ref_recursor(
                    item, class_mapping, entity_pool_raw, entity_pool_deserialized
                )
                if item.__class__ not in (bool, int, float, str)
                else item
                # for performance: do not recurse on simple primitives
                for item in data["__contents"]
            )

        if type_name_short == "complex128":
            return np.complex128(data["real"] + data["imag"] * 1j)
        if type_name_short == "complex":
            return complex(data["real"] + data["imag"] * 1j)
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
    class_mapping[XarrayDataArrayDeserializer._type_] = XarrayDataArrayDeserializer
    class_mapping[XarrayDatasetDeserializer._type_] = XarrayDatasetDeserializer
    entity_pool = {}

    for entity_list in data.get("entities", {}).values():
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
    if _issubclass(cls, list):
        return [deserialize_from_dict(item, class_mapping) for item in data]
    if _issubclass(cls, Mapping):
        type_name = data.get("__type")
        if type_name == "set":
            return {
                deserialize_from_dict(item, class_mapping)
                for item in data.get("__content")
            }
        if type_name == "tuple":
            return tuple(
                deserialize_from_dict(item, class_mapping)
                for item in data.get("__content")
            )
        out_mapping = {}
        for k, v in data.items():
            if k != "__type":
                out_mapping[k] = deserialize_from_dict(v, class_mapping)
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
    elif _issubclass(cls, NumpyArrayRepr):
        nparray_dir = get_fields_and_slots(data)
        mapping = {a_or_s: getattr(data, a_or_s) for a_or_s in nparray_dir}
        return NumpyArrayRepr(**mapping)
    else:
        return data
