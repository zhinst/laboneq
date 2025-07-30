# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import inspect
import types
import typing
from collections.abc import Iterable
from enum import Enum
from typing import Type, TypeVar

import attrs
import numpy
from cattrs import Converter
from cattrs.gen import make_dict_unstructure_fn
from cattrs.preconf.orjson import make_converter

from laboneq.serializers.core import from_dict
from laboneq.serializers.implementations.numpy_array import NumpyArraySerializer

# Common types

# cattrs does not work with typing.TypeAlias, which is the true identity of
# np.typing.ArrayLike. Ideally, np.typing.ArrayLike should be as follows:
# ArrayLike_Model = Union[
#     numpy._typing._nested_sequence._NestedSequence[
#         Union[bool, int, float, complex, str, bytes]
#     ],
#     numpy._typing._array_like._Buffer,
#     numpy._typing._array_like._SupportsArray[numpy.dtype[Any]],
#     numpy._typing._nested_sequence._NestedSequence[
#         numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]
#     ],
#     complex,
#     bytes,
# ]
# But we will be more practical and use the following:
ArrayLike_Model = numpy.ndarray | list[bool | int | float | str | bytes]


def _structure_arraylike(obj, _):
    """Structure a numpy array or a list of numbers."""
    if isinstance(obj, dict):
        return from_dict(obj)
    return numpy.asarray(obj)


def _unstructure_arraylike(obj):
    """Unstructure a numpy array or a list of numbers."""
    if isinstance(obj, numpy.ndarray):
        return NumpyArraySerializer.to_json_dict(obj)
    return obj


def _unstructure_complex_or_np_numbers(obj):
    if isinstance(obj, (numpy.complexfloating, complex)):
        return [obj.real, obj.imag]
    return obj


def _structure_complex_or_np_numbers(obj, _):
    if isinstance(obj, list) and len(obj) == 2:
        return numpy.complex128(obj[0], obj[1])
    return obj


# Supporting functions for serialization


def unstructure_union_generic_type(
    obj, types: Iterable[Type], converter: Converter
) -> None:
    """
    Unstructure the object using the first type in the list that matches the object type.
    Args:
        obj: The object to unstructure.
        types: The types to check against.
        converter: The cattr converter that must know how to unstructure the objects.
    """
    for t in types:
        if type(obj) is t._target_class:
            return {"_type": type(obj).__name__, **converter.unstructure(obj, t)}
    raise ValueError(
        f"Unsupported type: {type(obj).__name__} when unstructuring Union[{types}]"
    )


def structure_union_generic_type(
    d, types: Iterable[Type], converter: Converter
) -> None:
    """
    Structure the object using the first type in the list that matches the object type.
    Args:
        d: The dict to structure.
        types: The types to check against.
        converter: The cattr converter that must know how to structure the objects.
    """
    dtype = d.get("_type", None)
    if dtype is None:
        raise ValueError(f"Missing the _type field when structuring Union[{types}]")
    for t in types:
        if dtype == t._target_class.__name__:
            return converter.structure(d, t)


def structure(data: dict, cls: type, converter: Converter):
    """
    Structure a dictionary into an instance of the given class.
    Especially used for attrs-based models that have their attributes
    required to be converted to an instance with type specified in `_target_class` attribute.
    """
    attributes = attrs.fields_dict(cls)
    se = {}
    for k, v in data.items():
        if k == "_type" or k == "$ref":
            # _type field was used to manually structure the union type.
            # It is no longer needed when we let cattrs structure the object automatically.
            # ref field is used for caching and is already processed by the cache methods in ObjectCache.
            continue
        if k not in attributes:
            raise ValueError(f"Invalid attribute {k} for class {cls}")
        # to handle private attributes in attrs models
        key_name = k[1:] if k.startswith("_") else k

        # cattrs issue: https://github.com/python-attrs/cattrs/issues/656
        # Specifically handle the case of when value is a float or int and
        # the type is float or Union with float.
        # This cattrs issue was fixed and will be released in 25.2.
        # This work-around could perhaps be removed after we migrating to 25.2
        if typing.get_origin(attributes[k].type) in (
            typing.Union,
            types.UnionType,
            typing.Optional,
        ):
            if float in typing.get_args(attributes[k].type) and isinstance(
                v, (int, float)
            ):
                se[key_name] = v
                continue
        se[key_name] = converter.structure(v, attributes[k].type)
    de = cls._target_class(**se)
    return de


def _predicate(cls):
    return lambda obj: obj is cls


def register_models(converter: Converter, models: Iterable) -> None:
    for model in models:
        if issubclass(model, Enum):
            # Register the enum models
            converter.register_structure_hook(
                model, lambda d, cls: cls._target_class.value(d)
            )
            continue
        if "_unstructure" in model.__dict__:
            if hasattr(model, "__cache_serializer__"):
                converter.register_unstructure_hook_func(
                    _predicate(model),
                    model.__cache_serializer__.cache_unstructure(model._unstructure),
                )
            else:
                converter.register_unstructure_hook_func(
                    _predicate(model), model._unstructure
                )
        else:
            if hasattr(model, "__cache_serializer__"):
                converter.register_unstructure_hook(
                    model,
                    model.__cache_serializer__.cache_unstructure(
                        make_dict_unstructure_fn(model, converter)
                    ),
                )
            else:
                converter.register_unstructure_hook(
                    model, make_dict_unstructure_fn(model, converter)
                )

        if "_structure" in model.__dict__:
            if hasattr(model, "__cache_serializer__"):
                converter.register_structure_hook_func(
                    _predicate(model),
                    model.__cache_serializer__.cache_structure(model._structure),
                )
            else:
                converter.register_structure_hook_func(
                    _predicate(model), model._structure
                )
        else:
            if hasattr(model, "__cache_serializer__"):
                cache_func = model.__cache_serializer__.cache_structure(structure)
                converter.register_structure_hook(
                    model,
                    lambda d, cls, cache_func=cache_func: cache_func(d, cls, converter),
                )
            else:
                converter.register_structure_hook(
                    model, lambda d, cls: structure(d, cls, converter)
                )


def collect_models(module_models) -> frozenset:
    """Collect all attrs models for serialization."""
    subclasses = []
    for _, cls in inspect.getmembers(module_models):
        if hasattr(cls, "_target_class"):
            subclasses.append(cls)

    return frozenset(subclasses)


def unstructure_enum(obj) -> str:
    """
    Unstructure an Enum object to its value.
    Required for cattrs >=25.1 to handle Enum objects correctly.

    Args:
        obj: The Enum object to unstructure.
        cls: The Enum class.
    Returns:
        The value of the Enum object.
    """
    # Intentionally not checking the type here for performance reasons.
    # cattrs will call this function only for Enum objects.
    return obj.value


E = TypeVar("E", bound=Enum)


def structure_enum(d: str, cls: Type[E]) -> E:
    """
    Structure a string to an Enum object.
    Required for cattrs >=25.1 to handle Enum objects correctly.

    Args:
        d: The string to structure.
        cls: The Enum class.
    Returns:
        The Enum object.
    """
    # Intentionally not checking the type here for performance reasons.
    return cls(d)


def make_laboneq_converter() -> Converter:
    converter = make_converter()

    # TODO: Remove the hooks for ArrayLike_Model and only
    # register the hooks for numpy.ndarray and let the list of numbers to
    # be passed as is.
    converter.register_structure_hook(ArrayLike_Model, _structure_arraylike)
    converter.register_unstructure_hook(
        ArrayLike_Model,
        _unstructure_arraylike,
    )
    converter.register_structure_hook(numpy.ndarray, _structure_arraylike)
    converter.register_unstructure_hook(
        numpy.ndarray,
        _unstructure_arraylike,
    )

    converter.register_unstructure_hook(complex, _unstructure_complex_or_np_numbers)
    converter.register_structure_hook(complex, _structure_complex_or_np_numbers)
    converter.register_unstructure_hook(
        numpy.number, _unstructure_complex_or_np_numbers
    )
    converter.register_structure_hook(numpy.number, _structure_complex_or_np_numbers)
    converter.register_unstructure_hook(
        complex | numpy.number, _unstructure_complex_or_np_numbers
    )
    converter.register_structure_hook(
        complex | numpy.number, _structure_complex_or_np_numbers
    )

    converter.register_unstructure_hook(Enum, unstructure_enum)
    converter.register_structure_hook(Enum, structure_enum)

    return converter
