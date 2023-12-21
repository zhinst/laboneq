# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from functools import lru_cache
from unsync import unsync
from typing import Awaitable, TypeVar, Callable, Any
import asyncio


T = TypeVar("T")


# Required to keep sync interface callable from Jupyter Notebooks
# See https://blog.jupyter.org/ipython-7-0-async-repl-a35ce050f7f7
@lru_cache(maxsize=None)
def _is_event_loop_running() -> bool:
    try:
        asyncio.get_running_loop()
        # Running event loop detected
        return True
    except RuntimeError:
        # No event loop is running
        return False


def run_async(func: Callable[[Any], Awaitable[T]], *args, **kwargs) -> T:
    """Run callable asynchronous object synchronously.

    Args:
        func: Asynchronous callable to be called with `*args` and `*kwargs`
    """
    if _is_event_loop_running():
        return unsync(func)(*args, **kwargs).result()
    else:
        return asyncio.run(func(*args, **kwargs))
