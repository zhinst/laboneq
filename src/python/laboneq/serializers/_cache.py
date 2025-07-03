# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable, Generator, Generic, TypeVar

from laboneq.dsl.experiment.pulse import Pulse
from laboneq.dsl.experiment.section import Section

_T = TypeVar("_T")
_C = TypeVar("_C", bound=type)
_F = TypeVar("_F", bound=Callable[..., Any])


class _ContextObjectCacheStorage(threading.local):
    _object_cache: ObjectCache | None = None


_contexts_pulse = _ContextObjectCacheStorage()
_contexts_section = _ContextObjectCacheStorage()


@contextmanager
def create_caches() -> Generator[ObjectCache, None, None]:
    with (
        PulseCache.create_object_cache(),
        SectionCache.create_object_cache(),
    ):
        yield None


class ObjectCache(Generic[_T]):
    """A cache for objects that allows to store and retrieve them by a unique key.
    Each key, by default, is a string that starts with a prefix (e.g., "o" for objects).
    and the key is associated with the object ID.

    Usage: decorate the class with `@ObjectCache.cache` to enable caching for serialization and deserialization.
    We are currently supporting caching for `Pulse` and `Section` objects, with prefixes "p" and "s", respectively.

    To use the cache on a new object type, you can subclass `ObjectCache` and set the `_contexts` and `_key_prefix` attributes accordingly.
    The `_contexts` attribute should be set to the appropriate `_ContextObjectCacheStorage` instance.
    """

    _storage: dict[str, _T]
    _id_map: dict[int, str]
    _object_id: int
    _contexts: _ContextObjectCacheStorage = _ContextObjectCacheStorage()
    _key_prefix: str = "o"

    def __init__(self) -> None:
        self._storage = {}
        self._id_map = {}
        self._object_id = 0

    @classmethod
    @contextmanager
    def create_object_cache(cls) -> Generator["ObjectCache", None, None]:
        if cls._contexts._object_cache is not None:
            raise RuntimeError("Object cache was already initialized.")
        object_cache = cls()
        cls._contexts._object_cache = object_cache
        try:
            yield object_cache
        finally:
            cls._contexts._object_cache = None

    @classmethod
    def get_object_cache(cls) -> "ObjectCache":
        if cls._contexts._object_cache is None:
            raise RuntimeError("Object cache was not initialized.")
        return cls._contexts._object_cache

    def get(self, key: str) -> _T | None:
        """Retrieve the pulse from the cache with the given key.

        Arguments:
            key: The key to look up.

        Return:
            None if the object is not in the cache,
            otherwise the stored pulse object.
        """

        return self._storage.get(key)

    def get_key(self, obj: _T) -> str | None:
        """Retrieve the key used for the given pulse.

        Arguments:
            pulse: The pulse to look up.

        Returns:
            None if the pulse is not present, otherwise
            the key the pulse is stored under.
        """

        return self._id_map.get(id(obj))

    def add(self, obj: _T, key: str | None = None) -> str:
        """Add the object to the cache and return the key.

        Arguments:
            pulse: The pulse to add.

        Returns:
            The key the pulse was stored under.
        """

        if key is None:
            key = f"{self._key_prefix}{self._object_id}"
            self._object_id += 1
        self._storage[key] = obj
        self._id_map[id(obj)] = key
        return key

    @classmethod
    def _cache_unstructure(cls, obj: _T, func, args, kwargs) -> dict:
        """Cache the unstructured object and return its key."""
        object_cache = cls.get_object_cache()
        ref = object_cache.get_key(obj)
        if ref is None:
            ref = object_cache.add(obj)
            uncached_result = func(*args, **kwargs)
            uncached_result["$ref"] = ref
            return uncached_result
        return {"$ref": ref}

    @classmethod
    def cache_unstructure(cls, func: _F) -> _F:
        """Add caching support to a function that unstructures an object."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            o = args[0]
            return cls._cache_unstructure(o, func, args, kwargs)

        return wrapper

    @classmethod
    def cache_structure(cls, func: _F) -> _F:
        """Add caching support to a function that structures an object."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            data = args[0]
            if "$ref" not in data:
                return func(*args, **kwargs)
            return cls._cache_structure(data, func, args, kwargs)

        return wrapper

    @classmethod
    def _cache_structure(cls, data: dict, func: Callable, args, kwargs) -> Any:
        """Unstructure an object and cache it if it is not already cached."""
        object_cache = cls.get_object_cache()
        obj = object_cache.get(data["$ref"])
        if obj is None:
            obj = func(*args, **kwargs)
            object_cache.add(obj, key=data["$ref"])
        return obj

    @classmethod
    def cache(cls, decorated_class: _C) -> _C:
        """Decorate a model class to use the cache for serialization and deserialization."""
        decorated_class.__cache_serializer__ = cls
        return decorated_class


class PulseCache(ObjectCache[Pulse]):
    _contexts: _ContextObjectCacheStorage = _contexts_pulse
    _key_prefix: str = "p"


class SectionCache(ObjectCache[Section]):
    _contexts: _ContextObjectCacheStorage = _contexts_section
    _key_prefix: str = "s"
