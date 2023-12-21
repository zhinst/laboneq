# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laboneq.controller.communication import ServerQualifier
    from laboneq.controller.devices.device_zi import DeviceQualifier


async def create_device_kernel_session(
    *, server_qualifier: ServerQualifier, device_qualifier: DeviceQualifier
):
    return None  # TODO(2K): stub, will return the real async api kernel session


@asynccontextmanager
async def gather_and_apply(func):
    awaitables = []
    yield awaitables
    results = await asyncio.gather(*awaitables)
    await func([value for values in results for value in values])
