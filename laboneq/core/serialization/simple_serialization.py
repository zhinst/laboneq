# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import base64
import dataclasses
import functools
import importlib
import inspect
import logging
from collections.abc import Iterable
from collections.abc import Mapping
from enum import Enum
from io import BytesIO
from typing import Optional

import networkx as nx
import numpy
import numpy as np
from numpy.typing import ArrayLike
from sortedcontainers import SortedDict

ID_KEY = "__id"


@dataclasses.dataclass(init=False)
class NumpyArrayRepr:
    array_data: Optional[ArrayLike] = dataclasses.field(default=None)
    real_data: Optional[ArrayLike] = dataclasses.field(default=None)
    complex_data: Optional[ArrayLike] = dataclasses.field(default=None)
    binary_npz: Optional[str] = dataclasses.field(default=None)

    def __new__(
        cls, array_data=None, real_data=None, complex_data=None, binary_npz=None
    ):
        # deserialize
        if binary_npz is not None:
            input_buffer = BytesIO(base64.b64decode(binary_npz.encode("ascii")))
            loaded_npz = np.load(input_buffer)
            return loaded_npz[loaded_npz.files[0]]

        if real_data is not None:
            return np.array(real_data)

        if complex_data is not None:
            return np.array(complex_data).astype(complex)

        # serialize
        return super().__new__(cls)

    def __init__(
        self, array_data=None, real_data=None, complex_data=None, binary_npz=None
    ):
        assert array_data is not None
        assert real_data is None and complex_data is None and binary_npz is None
        self.array_data = None
        self.real_data = None
        self.complex_data = None
        self.binary_npz = None

        if max(np.shape(array_data)) > 400:
            output_buffer = BytesIO()
            np.savez_compressed(output_buffer, array_data)
            self.binary_npz = base64.b64encode(output_buffer.getbuffer()).decode(
                "ascii"
            )
        else:
            if np.isrealobj(array_data):
                self.real_data = array_data.tolist()
            else:
                self.complex_data = np.frompyfunc(str, 1, 1)(array_data).tolist()


# TODO(2K): Replace this simplified approach with the portable and secure one
class CallableRepr:
    # __new__ does the actual deserialization job, receiving dict elements as arguments
    def __new__(cls, name: str, numpy_name: Optional[str], body: str):
        namespace = {numpy_name: numpy} if numpy_name is not None else {}
        exec(body, namespace)
        return namespace[name]

    # __init__ signature is used to access dict elements on deserialization
    def __init__(self, name: str, numpy_name: Optional[str], body: str):
        pass


def serialize_callable(to_serialize):
    # Some validations, but not necessarily exhaustive and portable,
    # in particular missing imports for type hints will not be detected,
    # but will fail on deserialization.
    cv = inspect.getclosurevars(to_serialize)
    numpy_name: Optional[str] = None
    if len(cv.globals) >= 1:
        name, value = next(iter(cv.globals.items()))
        if value is numpy:
            numpy_name = name
        if value is not numpy or len(cv.globals) > 1:
            raise RuntimeError(
                f"User function {to_serialize.__name__} must not use any globals except 'numpy', but is using {[k for k in cv.globals]}"
            )
    if len(cv.nonlocals) > 0:
        raise RuntimeError(
            f"User function {to_serialize.__name__} must not use any non-locals, but is using {[k for k in cv.nonlocals]}"
        )
    return {
        "__type": CallableRepr.__name__,
        "name": to_serialize.__name__,
        "numpy_name": numpy_name,
        "body": inspect.getsource(to_serialize),
    }


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
        return None
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

    if callable(to_serialize):
        return serialize_callable(to_serialize)

    if _issubclass(cls, str) or _issubclass(cls, float) or _issubclass(cls, int):
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
    if hasattr(to_serialize, "__dict__"):
        is_object = True
        item_is_entity, item_entity_class = is_entity_class(
            to_serialize.__class__, entity_classes
        )
        if item_is_entity:
            if to_serialize.uid is None:
                raise RuntimeError(f"to_serialize has uid of None, {to_serialize}")

        mapping = to_serialize.__dict__
        sub_dict["__type"] = short_typename(to_serialize)

    elif _issubclass(cls, Mapping):
        mapping = to_serialize

    for k, v in mapping.items():
        outkey = k
        outvalue = v

        item_is_entity, item_entity_class = is_entity_class(v.__class__, entity_classes)
        if k[0] == "_" and is_object:
            if k[1:] in dir_list:
                outkey = k[1:]
                outvalue = getattr(to_serialize, outkey)
            else:
                continue
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
        else:
            out_cls = outvalue.__class__

            if _issubclass(out_cls, np.ndarray):
                sub_dict[outkey] = serialize_to_dict_with_entities(
                    NumpyArrayRepr(array_data=outvalue),
                    entity_classes,
                    entities_collector,
                    emit_enum_types,
                )

            elif _issubclass(out_cls, Mapping):
                sub_dict[outkey] = serialize_to_dict_with_entities(
                    v, entity_classes, entities_collector, emit_enum_types
                )
            elif _issubclass(out_cls, Iterable) and not _issubclass(out_cls, str):
                sub_dict[outkey] = []
                index = 0
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

                    index += 1
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


def calculate_reference_graph(
    data, class_mapping, entity_classes, reference_graph, entity_map, current_entity
):
    cls = data.__class__
    if _issubclass(cls, Mapping):
        if ("$ref" in data) and ("__entity_type" in data):
            if current_entity is not None:
                entity_key = (current_entity, (data["$ref"], data["__entity_type"]))
            else:
                entity_key = (("__ROOT__", ""), (data["$ref"], data["__entity_type"]))
            reference_graph.append(entity_key)
            return None
        else:
            type_name = data.get("__type")
            new_entity = current_entity

            if type_name is not None:
                type_name_short = type_name.split(".")[-1]
                mapped_class = class_mapping.get(type_name_short)
                if mapped_class is None:
                    raise Exception(
                        f"Class {type_name_short} / {type_name} not found, known classes are {list(class_mapping.keys())}"
                    )
                current_class_is_entity_class, entity_class = is_entity_class(
                    mapped_class, entity_classes
                )
                if current_class_is_entity_class:
                    entity_class_name_short = full_classname(entity_class).split(".")[
                        -1
                    ]
                    new_entity = (data["uid"], entity_class_name_short)
                    entity_map[new_entity] = data

            for k, v in data.items():
                if k != "__type":
                    calculate_reference_graph(
                        v,
                        class_mapping,
                        entity_classes,
                        reference_graph,
                        entity_map,
                        new_entity,
                    )

    elif _issubclass(cls, Iterable) and not _issubclass(cls, str):
        for item in data:
            calculate_reference_graph(
                item,
                class_mapping,
                entity_classes,
                reference_graph,
                entity_map,
                current_entity,
            )


def deserialize_from_dict_with_ref_recursor(data, class_mapping, entity_collector):
    cls = data.__class__
    if _issubclass(cls, Mapping):
        if ("$ref" in data) and ("__entity_type" in data):
            return entity_collector[data["__entity_type"]][data["$ref"]]
        else:
            out_mapping = {}
            for k, v in data.items():
                if k != "__type":
                    child = deserialize_from_dict_with_ref_recursor(
                        v, class_mapping, entity_collector
                    )
                    out_mapping[k] = child
            type_name = data.get("__type")
            if type_name is not None:
                type_name_short = type_name.split(".")[-1]
                mapped_class = class_mapping.get(type_name_short)
                if mapped_class is None:
                    raise Exception(
                        f"Class {type_name_short} / {type_name} not found, known classes are {list(class_mapping.keys())}"
                    )
                constructed_object = construct_object(out_mapping, mapped_class)
                return constructed_object
            else:
                return out_mapping
    elif _issubclass(cls, Iterable) and not _issubclass(cls, str):
        out_list = [
            deserialize_from_dict_with_ref_recursor(
                item, class_mapping, entity_collector
            )
            for item in data
        ]
        return out_list
    else:
        return data


def deserialize_from_dict_with_ref(data, class_mapping, entity_classses, entity_map):
    class_mapping[NumpyArrayRepr.__name__] = NumpyArrayRepr
    class_mapping[CallableRepr.__name__] = CallableRepr
    entity_map = {}
    reference_graph = []
    calculate_reference_graph(
        data, class_mapping, entity_classses, reference_graph, entity_map, None
    )
    entity_map[("__ROOT__", "")] = next(v for k, v in data.items() if k != "entities")
    reference_graph_nx = nx.DiGraph()
    reference_graph_nx.add_edges_from(reference_graph)

    g_entity_collector = {}
    g_root_object = None
    for entity_key in list(reversed(list(nx.topological_sort(reference_graph_nx)))):
        if entity_key[0] != "__ROOT__":
            if entity_key[1] not in g_entity_collector:
                g_entity_collector[entity_key[1]] = {}
            cur_object = deserialize_from_dict_with_ref_recursor(
                entity_map[entity_key], class_mapping, g_entity_collector
            )
            g_entity_collector[entity_key[1]][entity_key[0]] = cur_object
        else:
            g_root_object = deserialize_from_dict_with_ref_recursor(
                entity_map[entity_key], class_mapping, g_entity_collector
            )
    if g_root_object is None:
        g_root_object = deserialize_from_dict_with_ref_recursor(
            entity_map[("__ROOT__", "")], class_mapping, g_entity_collector
        )

    return g_root_object


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
