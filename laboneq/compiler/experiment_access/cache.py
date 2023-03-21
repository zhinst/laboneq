# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from functools import wraps


def memoize_method(method):
    """Decorator to memoize a method of a class.

    Stores the cache in the instance, so it does not hash `self`.
    """
    cache_name = f"__{method.__name__}_cache"

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        cache = self.__dict__.setdefault(cache_name, {})

        key = (args, frozenset(kwargs.items()))
        if key not in cache:
            cache[key] = method(self, *args, **kwargs)
        return cache[key]

    return wrapper
