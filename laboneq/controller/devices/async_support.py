# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Callable, Coroutine, TypeVar
from laboneq.controller.devices.device_utils import NodeCollector

from laboneq.controller.devices.zi_emulator import EmulatorState

if TYPE_CHECKING:
    from laboneq.controller.communication import ServerQualifier
    from laboneq.controller.devices.device_zi import DeviceQualifier


ASYNC_DEBUG_MODE = False


async def create_device_kernel_session(
    *,
    server_qualifier: ServerQualifier,
    device_qualifier: DeviceQualifier,
    emulator_state: EmulatorState | None,
) -> None:
    return None  # TODO(2K): stub, will return the real async api kernel session


U = TypeVar("U")


async def _gather(*args: Coroutine[Any, Any, U]) -> list[U]:
    if ASYNC_DEBUG_MODE:
        return [await arg for arg in args]
    return await asyncio.gather(*args)


@asynccontextmanager
async def gather_and_apply(func: Callable[[list[U]], Coroutine[Any, Any, None]]):
    awaitables: list[Coroutine[Any, Any, U]] = []
    yield awaitables
    await func(await _gather(*awaitables))


async def set_parallel(api: Any, nodes: NodeCollector):
    return  # TODO(2K): stub


async def get_raw(api: Any, path: str) -> dict[str, Any]:
    return {}  # TODO(2K): stub
