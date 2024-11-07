# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.workflow.typing import SimpleDict
import numpy as np


class DeduplicationCache:
    """A cache for deduplicating serialization of objects.

    The cache consists of two dictionaries: One is mapping the id of an
    object to a tuple of the object and the serialized object; the second
    maps a hash of the serialized string to the serialized object. The cache
    is used to avoid storing the same object multiple times.
    """

    def __init__(self) -> None:
        self._by_id: dict[int, tuple[object, SimpleDict, object]] = {}
        self._by_string: dict[int, object] = {}

    def get_from_object(self, obj: object, options: SimpleDict) -> object | None:
        """Get the serialized object from the cache via its object id."""
        cached = self._by_id.get(id(obj))
        if cached is not None:
            cached_obj, cached_options, cached_files = cached
            if isinstance(obj, np.ndarray) and isinstance(cached_obj, np.ndarray):
                is_equal = np.array_equal(cached_obj, obj)
            else:
                is_equal = cached_obj == obj
            if is_equal and cached_options == options:
                return cached_files
        return None

    def store_object(self, obj: object, options: SimpleDict, files: object) -> None:
        self._by_id[id(obj)] = (obj, options, files)
