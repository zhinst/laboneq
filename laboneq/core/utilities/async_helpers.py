# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from unsync import unsync
from typing import Awaitable, TypeVar, Callable, Any
import asyncio


T = TypeVar("T")


def run_sync(func: Callable[[Any], Awaitable[T]], *args, **kwargs) -> T:
    """Run callable asynchronous object synchronously.

    Args:
        func: Asynchronous callable to be called with `*args` and `*kwargs`
    """
    try:
        asyncio.get_running_loop()
        # Running event loop detected
        need_nesting = True
    except RuntimeError:
        # No event loop is running
        need_nesting = False

    if need_nesting:
        return unsync(func)(*args, **kwargs).result()
    else:
        return asyncio.run(func(*args, **kwargs))
