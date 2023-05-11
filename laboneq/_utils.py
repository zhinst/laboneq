# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""General utility functions for development."""
import functools
from typing import Any


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
