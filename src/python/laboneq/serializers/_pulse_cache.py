# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from contextlib import contextmanager
import threading
from typing import Generator
from laboneq.dsl.experiment.pulse import PulseSampled


class _ContextPulseCacheStorage(threading.local):
    _pulse_cache: PulseSampledCache | None = None


_contexts = _ContextPulseCacheStorage()


class PulseSampledCache:
    _storage: dict[str, PulseSampled]
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
        """Retrieve the pulse from the cache with the given key.

        Arguments:
            key: The key to look up.

        Return:
            None if the object is not in the cache,
            otherwise the stored pulse object.
        """
        return self._storage.get(key)

    def get_key(self, pulse: PulseSampled) -> str | None:
        """Retrieve the key used for the given pulse.

        Arguments:
            pulse: The pulse to look up.

        Returns:
            None if the pulse is not present, otherwise
            the key the pulse is stored under.
        """
        return self._id_map.get(id(pulse))

    def add(self, pulse: PulseSampled, key: str | None = None) -> tuple[str, bool]:
        """Add the object to the cache and return the key.

        Arguments:
            pulse: The pulse to add.

        Returns:
            The key the pulse was stored under.
        """
        if key is None:
            key = f"p{self._pulse_id}"
            self._pulse_id += 1
        self._storage[key] = pulse
        self._id_map[id(pulse)] = key
        return key
