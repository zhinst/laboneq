# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import inspect
from collections.abc import Iterable
from typing import Type, Union

import attrs
import numpy
from cattrs import Converter
from cattrs.gen import make_dict_structure_fn, make_dict_unstructure_fn
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
ArrayLike_Model = Union[
    numpy.ndarray,
    list[Union[bool, int, float, str, bytes]],
]


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


@attrs.define
class NumericModel:
    """
    A model representing numeric values: int, float, complex
    and numpy numbers.

    Defines an attrs-based model to enable `cattrs` to **automatically**
    serialize and deserialize the following types:
    - Unions of Python numeric types and NumPy number types
    - Unions of numeric types and other attrs-based models
    """

    @classmethod
    def _unstructure(cls, obj):
        if isinstance(obj, (numpy.complexfloating, complex)):
            return [obj.real, obj.imag]
        return obj

    @classmethod
    def _structure(cls, obj, _):
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


def register_models(converter, models: Iterable) -> None:
    # Use predicate hooks to avoid type evaluation problems
    # for complicate types in Python 3.9.
    # TODO: When dropping support for Python 3.9,
    # Remove registering predicate hooks (register_(un)structure_hook_func)
    # and use dispatcher hooks instead (register_(un)structure_hook)
    # See the docstring of this module for more details.
    def _predicate(cls):
        return lambda obj: obj is cls

    for model in models:
        if hasattr(model, "_unstructure"):
            converter.register_unstructure_hook_func(
                _predicate(model), model._unstructure
            )
        else:
            converter.register_unstructure_hook(
                model, make_dict_unstructure_fn(model, converter)
            )
        if hasattr(model, "_structure"):
            converter.register_structure_hook_func(_predicate(model), model._structure)
        else:
            converter.register_structure_hook(
                model, make_dict_structure_fn(model, converter)
            )


def collect_models(module_models) -> frozenset:
    """Collect all attrs models for serialization. Enum models are
    already registered in the pre-configured converter."""
    subclasses = []
    for _, cls in inspect.getmembers(module_models):
        if attrs.has(cls) and hasattr(cls, "_target_class"):
            subclasses.append(cls)

    return frozenset(subclasses)


def make_laboneq_converter() -> Converter:
    converter = make_converter()

    converter.register_structure_hook(ArrayLike_Model, _structure_arraylike)
    converter.register_unstructure_hook(
        ArrayLike_Model,
        _unstructure_arraylike,
    )

    converter.register_unstructure_hook(NumericModel, NumericModel._unstructure)
    converter.register_structure_hook(NumericModel, NumericModel._structure)

    return converter
