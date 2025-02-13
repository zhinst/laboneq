# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Base classes and utilities for LabOne Q serialization and deserialization."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import orjson
from laboneq._version import get_version
from laboneq.serializers.serializer_registry import (
    serializer_registry,
)

if TYPE_CHECKING:
    from laboneq.serializers.serializer_registry import SerializerRegistry
    from laboneq.serializers.types import (
        DeserializationOptions,
        JsonSerializableType,
        SerializationOptions,
    )


def _registry(
    options: SerializationOptions | DeserializationOptions | None,
) -> SerializerRegistry:
    if options is None:
        return serializer_registry
    return options.serializer_registry or serializer_registry


def _is_generic_type(obj: object) -> bool:
    """Check if type can be handled by normal json serializers."""
    # TODO: Could we utilize orjson/json.dumps() to these by default?
    return obj is None or isinstance(
        obj, (int, float, bool, str, np.integer, np.ndarray)
    )


def _serialize_object(
    obj: object, options: SerializationOptions | None = None, only_public=False
) -> JsonSerializableType:
    # TODO: Remove check and return if no serialize is found. Let orjson.dumps() handle
    #       simple types to avoid manual mapping.
    if _is_generic_type(obj):
        return obj
    registry = _registry(options)
    serializer = registry[type(obj)]
    if serializer is None:
        raise RuntimeError(
            f"No serializer available for object of type {type(obj).__module__}.{type(obj).__name__}."
        )
    if only_public and not registry.is_public(type(obj)):
        raise RuntimeError(
            f"No serializer publicly available for objects of type {type(obj).__module__}.{type(obj).__name__}."
        )
    return serializer.to_dict(obj, options)


def import_cls(full_name: str) -> type:
    """Attempt to import a class by name."""
    module_name, class_name = full_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ImportError(
            f"Failed to find {class_name!r} in module {module_name!r}."
        ) from None
    return cls


def to_dict(
    obj: object, options: SerializationOptions | None = None
) -> JsonSerializableType:
    """Store an object to a dict with only basic datatypes."""
    serialized = _serialize_object(obj, options, only_public=True)
    if isinstance(serialized, dict) and "__serializer__" in serialized:
        serialized["__creator__"] = ["laboneq", get_version()]
    return serialized


def from_dict(
    data: JsonSerializableType, options: DeserializationOptions | None = None
) -> object:
    """Load an object from a dict with basic datatypes."""
    if _is_generic_type(data):
        return data
    if isinstance(data, list):
        return [from_dict(d, options) for d in data]
    if "__serializer__" not in data:
        return {k: from_dict(v, options) for k, v in data.items()}
    serializer_id = data["__serializer__"]
    if not isinstance(serializer_id, str):
        raise ValueError(f"Serializer ID {serializer_id} must be of type str.")
    serializer = _registry(options)[serializer_id]
    if serializer is not None:
        return serializer.from_dict(data, options)
    # Try to load the serializer from the module
    try:
        serializer = import_cls(serializer_id)
    except ImportError as exc:
        raise ValueError(f"Failed to import serializer {serializer_id}.") from exc
    return serializer.from_dict(data, options)


def _default(options: SerializationOptions | None = None):
    """ "Create a default callback from JSON serializer.

    The callback looks for serializer registry and calls `.to_json_dict()`
    on registered objects it encounters.
    """

    def to_json_dict_caller(obj: object) -> object:
        if (serializer := _registry(options)[type(obj)]) is not None:
            return serializer.to_json_dict(obj, options)
        raise TypeError()

    return to_json_dict_caller


def to_json(
    obj: object,
    options: SerializationOptions | None = None,
) -> bytes:
    """Serialize an object to JSON."""
    json_dict = to_dict(obj, options)
    return orjson.dumps(
        json_dict,
        option=orjson.OPT_SORT_KEYS | orjson.OPT_SERIALIZE_NUMPY,
        default=_default(options),
    )


def from_json(data: bytes, options: DeserializationOptions | None = None) -> object:
    """Deserialize an object from JSON."""
    d = orjson.loads(data)
    return from_dict(d, options)


def save(
    obj: object, filename: Path, options: SerializationOptions | None = None
) -> None:
    """Store an object to a file."""
    with open(filename, "wb") as f:
        f.write(to_json(obj, options))


def load(filename: Path, options: DeserializationOptions | None = None) -> object:
    """Load an object from a file."""
    with open(filename, "rb") as f:
        return from_json(f.read(), options)
