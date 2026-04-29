# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Base classes and utilities for LabOne Q serialization and deserialization."""

from __future__ import annotations

import importlib
import warnings
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import orjson
import yaml

from laboneq._version import get_version
from laboneq.serializers.base import VersionedClassSerializer
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


class SerializerFormat(Enum):
    JSON = "json"
    YAML = "yaml"


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


# Sentinel used to detect callers that omit the new keyword arguments.
# We cannot use None as the default for prefix_restrictions because None is a
# meaningful value (meaning "allow any module"), so we need a distinct object
# to distinguish "caller passed None explicitly" from "caller passed nothing".
_UNSET = object()


def import_cls(
    dotted_name: str,
    *,
    base_cls: type | None = None,
    prefix_restrictions: tuple[str, ...] | None = _UNSET,  # type: ignore[assignment]
) -> type:
    """Import a class by its dotted name and verify it is a subclass of the expected type.

    !!! version-changed "Changed in version 26.4.0"
        Both ``base_cls`` and ``prefix_restrictions`` are required. Omitting either
        is deprecated and will raise an error in version 26.10.0.

    Args:
        dotted_name: Dotted name of the form ``"<module>.<ClassName>"``, e.g.
            ``"mypackage.module.MyClass"``. The module path must be importable
            and the class must be a top-level attribute of that module —
            nested classes (whose ``__qualname__`` contains a dot) are not
            supported.
        base_cls: The class that the imported class must be a subclass of.
            Required; omitting it is deprecated.
        prefix_restrictions: Use this when ``dotted_name`` comes from an
            untrusted source (e.g. a serialized file) to restrict which modules
            may be imported — ``dotted_name`` must start with one of the listed
            prefixes or a ``ValueError`` is raised. Pass ``None`` to allow any
            module when the caller already controls the name. Required;
            omitting it is deprecated.
    """
    if base_cls is None or prefix_restrictions is _UNSET:
        missing = []
        if base_cls is None:
            missing.append("base_cls")
        if prefix_restrictions is _UNSET:
            missing.append("prefix_restrictions")
        missing_str = " and ".join(missing)
        warnings.warn(
            f"Calling import_cls() without explicit {missing_str} "
            "is deprecated and will be removed in version 26.10.0. "
            "Pass base_cls=<expected base class> to verify the imported class, and "
            "prefix_restrictions=('<your.package.',) to restrict which modules may "
            "be imported (or prefix_restrictions=None to allow any module).",
            FutureWarning,
            stacklevel=2,
        )
        if base_cls is None:
            base_cls = object
        if prefix_restrictions is _UNSET:
            prefix_restrictions = None
    if prefix_restrictions is not None and not any(
        dotted_name.startswith(p) for p in prefix_restrictions
    ):
        prefix_str = " or ".join(f"'{p}'" for p in prefix_restrictions)
        raise ValueError(
            f"{dotted_name!r} cannot be imported: "
            f"only modules starting with {prefix_str} are allowed."
        )
    module_name, class_name = dotted_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    try:
        cls = getattr(module, class_name)
    except AttributeError:
        raise ImportError(
            f"Failed to find {class_name!r} in module {module_name!r}."
        ) from None
    if not isinstance(cls, type):
        raise ValueError(
            f"Expected a class (subclass of {base_cls.__name__}), "
            f"but {class_name!r} in module {module_name!r} is not a class: got {cls!r}."
        )
    if not issubclass(cls, base_cls):
        raise ValueError(
            f"Expected a subclass of {base_cls.__name__}, "
            f"but {cls.__qualname__} (from {cls.__module__}) is not."
        )
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
    # Serializer not in registry — try importing it by name.
    # Restrict to laboneq packages to prevent import-time side effects from
    # crafted __serializer__ values in untrusted data.
    try:
        serializer = import_cls(
            serializer_id,
            base_cls=VersionedClassSerializer,
            prefix_restrictions=("laboneq.", "laboneq_zqcs."),
        )
    except ImportError as exc:
        raise ValueError(f"Failed to import serializer {serializer_id!r}.") from exc
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


def to_yaml(
    obj: object,
    options: SerializationOptions | None = None,
) -> str:
    """Serialize an object to YAML."""
    json_dict = to_dict(obj, options)
    return yaml.dump(json_dict, sort_keys=True)


def from_yaml(data: str, options: DeserializationOptions | None = None) -> object:
    """Deserialize an object from YAML."""
    d = yaml.safe_load(data)
    return from_dict(d, options)


def save(
    obj: object,
    filename: Path,
    options: SerializationOptions | None = None,
    format: SerializerFormat = SerializerFormat.JSON,
) -> None:
    """Store an object to a file."""
    with open(filename, "wb") as f:
        f.write(
            to_json(obj, options)
            if format == SerializerFormat.JSON
            else to_yaml(obj, options).encode("utf-8")
        )


def load(
    filename: Path,
    options: DeserializationOptions | None = None,
    format: SerializerFormat = SerializerFormat.JSON,
) -> object:
    """Load an object from a file."""
    with open(filename, "rb") as f:
        return (
            from_json(f.read(), options)
            if format == SerializerFormat.JSON
            else from_yaml(f.read().decode("utf-8"), options)
        )
