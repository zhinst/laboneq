# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from contextlib import contextmanager
import threading
from typing import Generator
from laboneq.dsl.experiment.pulse import PulseSampled
import weakref


class _ContextPulseCacheStorage(threading.local):
    _pulse_cache: PulseSampledCache | None = None


_contexts = _ContextPulseCacheStorage()


class PulseSampledCache:
    _storage: dict[str, weakref.ReferenceType[PulseSampled]]
    _id_map: dict[str, int]
    _pulse_id: float

    def __init__(self) -> None:
        self._storage = {}
        self._id_map = {}
        self._pulse_id = 0

    @classmethod
    @contextmanager
    def create_pulse_cache(cls) -> Generator:
        if _contexts._pulse_cache is not None:
            raise RuntimeError("Pulse cache was already initialized.")
        pulse_cache = cls()
        _contexts._pulse_cache = pulse_cache
        try:
            yield pulse_cache
        finally:
            _contexts._pulse_cache = None

    @classmethod
    def get_pulse_cache(cls) -> PulseSampledCache:
        if _contexts._pulse_cache is None:
            raise RuntimeError("Pulse cache was not initialized.")
        return _contexts._pulse_cache

    def get(self, key: str) -> PulseSampled | None:
        """Retrieve an object from the cache with the given key.
        Return None if the object is not in the cache or the weak reference is dead.
        Remove the entry if the weak reference is dead.
        """

        ref = self._storage.get(key)
        if ref is not None:
            obj = ref()
            if obj is not None:
                return obj
            else:
                # The weak reference is dead, remove the entry from the cache
                self._storage.pop(key, None)
        return None

    def add(self, value: PulseSampled) -> str:
        """Add the object to the cache and return the key, that
        could be used to retrieve the object from the cache.
        """
        _id = id(value)
        if _id in self._id_map:
            return self._id_map[_id]
        # Add the object to the cache
        key = f"p{self._pulse_id}"
        self._pulse_id += 1
        self._storage[key] = weakref.ref(value)
        self._id_map[key] = _id
        return key

    def __contains__(self, obj: PulseSampled) -> bool:
        return id(obj) in self._id_map.values()
