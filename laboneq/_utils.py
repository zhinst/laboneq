# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""General utility functions for development."""
import functools
from collections import defaultdict
from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable


def cached_method(maxsize: int = 128, typed=False) -> Any:
    """Cache method decorator.

    Arguments are forwarded to `functools.lru_cache`
    """

    def outer_wrapper(func):
        method_cache = f"__cache_cls_{func.__name__}"

        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            cached = getattr(self, method_cache, None)
            if cached is None:

                @functools.lru_cache(maxsize=maxsize, typed=typed)
                def cached_func(*args, **kwargs):
                    return func(self, *args, **kwargs)

                setattr(self, method_cache, cached_func)
                cached = cached_func
            return cached(*args, **kwargs)

        return wrapper

    return outer_wrapper


def ensure_list(obj):
    if not isinstance(obj, list):
        return [obj]
    return obj


def flatten(l: Iterable):
    """Flatten an arbitrarily nested list."""
    for el in l:
        if isinstance(el, Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


_iid_map: defaultdict[str, count] = defaultdict(count)


def id_generator(cat: str = "") -> str:
    """Incremental IDs for each category."""
    global _iid_map
    return f"_{cat}_{next(_iid_map[cat])}"


@dataclass
class UIDReference:
    """Reference to an object with an UID.

    Args:
        uid: UID of the referenced object.
    """

    uid: str
