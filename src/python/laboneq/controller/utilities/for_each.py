# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from collections.abc import Callable, Coroutine, Iterable
from typing import Any, TypeVar

from laboneq.controller.devices.async_support import _gather


Ret = TypeVar("Ret")


def for_each_sync(
    targets: Iterable[Any],
    method: Callable[..., Ret],
    *args,
    **kwargs,
) -> list[Ret]:
    """Call a method on each object returned by the targets iterator,
    if the object’s class matches the class that defines the method.
    """
    [class_name, method_name] = method.__qualname__.split(".", 1)
    method_class = method.__globals__[class_name]
    return [
        # To keep polymorph behavior, we use getattr on the object instance
        # to retrieve the actual method to call, using passed method only as
        # a reference to the method name.
        getattr(target, method_name)(*args, **kwargs)
        for target in targets
        if isinstance(target, method_class)
    ]


async def for_each(
    targets: Iterable[Any],
    method: Callable[..., Coroutine[Any, Any, Ret]],
    *args,
    **kwargs,
) -> list[Ret]:
    """Call an async method on each object returned by the targets iterator,
    if the object’s class matches the class that defines the method. Gather
    the results and return them as a list.
    """
    return await _gather(*for_each_sync(targets, method, *args, **kwargs))
