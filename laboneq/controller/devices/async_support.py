# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum, IntFlag
import json
from typing import TYPE_CHECKING, Any, Callable, Coroutine, TypeVar

import numpy as np
from laboneq.controller.devices.device_utils import NodeActionSet, NodeCollector

from laboneq.controller.devices.zi_emulator import EmulatorState, MockInstrument
from laboneq.controller.devices.zi_node_monitor import NodeMonitorBase

from labone import DataServer, Instrument
from labone.core import AnnotatedValue
from labone.core.subscription import DataQueue
from labone.core.shf_vector_data import preprocess_complex_shf_waveform_vector

if TYPE_CHECKING:
    from laboneq.controller.communication import ServerQualifier
    from laboneq.controller.devices.device_zi import DeviceQualifier


ASYNC_DEBUG_MODE = False


class _DeviceStatusFlag(IntFlag):
    """Device status codes."""

    CLEAR = 0
    NOT_YET_READY = 1 << 0
    FREE = 1 << 1
    IN_USE = 1 << 2
    FW_UPGRADE_USB = 1 << 3
    FW_UPGRADE_REQUIRED = 1 << 4
    FW_UPGRADE_AVAILABLE = 1 << 5
    FW_DOWNGRADE_REQUIRED = 1 << 6
    FW_DOWNGRADE_AVAILABLE = 1 << 7
    FW_UPDATE_IN_PROGRESS = 1 << 8
    UNKNOWN = 1 << 9


async def _get_device_statuses(
    dataserver: DataServer, serials: list[str]
) -> dict[str, _DeviceStatusFlag]:
    devices = json.loads((await dataserver.kernel_session.get("/zi/devices")).value)
    try:
        return {
            serial: _DeviceStatusFlag(devices[serial.upper()]["STATUSFLAGS"])
            for serial in serials
        }
    except KeyError as error:
        raise RuntimeError(f"Device {error} could not be found.") from error


def _check_dataserver_device_compatibility(statuses: dict[str, _DeviceStatusFlag]):
    errors = []
    for serial, flags in statuses.items():
        if _DeviceStatusFlag.FW_UPDATE_IN_PROGRESS in flags:
            raise ConnectionError(
                f"Device '{serial}' has update in progress. Wait for update to finish."
            )
        if _DeviceStatusFlag.FW_UPGRADE_AVAILABLE in flags:
            errors.append(
                f"Device '{serial}' has firmware upgrade available."
                "Please upgrade the device firmware."
            )
        if (
            _DeviceStatusFlag.FW_UPGRADE_REQUIRED in flags
            or _DeviceStatusFlag.FW_UPGRADE_USB in flags
        ):
            errors.append(
                f"Device '{serial}' requires firmware upgrade. "
                "Please upgrade the device firmware."
            )
        if _DeviceStatusFlag.FW_DOWNGRADE_AVAILABLE in flags:
            errors.append(
                f"Device '{serial}' has firmware downgrade available. "
                "Please downgrade the device firmware or update LabOne."
            )
        if _DeviceStatusFlag.FW_DOWNGRADE_REQUIRED in flags:
            errors.append(
                f"Device '{serial}' requires firmware downgrade. "
                "Please downgrade the device firmware or update LabOne."
            )
    if errors:
        raise RuntimeError(
            "LabOne and device firmware version compatibility issues were found.\n"
            + "\n".join(errors)
        )


async def async_check_dataserver_device_compatibility(
    host: str, port: int, serials: list[str]
):
    dataserver = await DataServer.create(host=host, port=port)
    statuses = await _get_device_statuses(dataserver, serials)
    _check_dataserver_device_compatibility(statuses)


async def create_device_kernel_session(
    *,
    server_qualifier: ServerQualifier,
    device_qualifier: DeviceQualifier,
    emulator_state: EmulatorState | None,
) -> Instrument:
    if emulator_state is not None:
        return MockInstrument(
            serial=device_qualifier.options.serial,
            emulator_state=emulator_state,
        )
    return await Instrument.create(
        serial=device_qualifier.options.serial,
        interface=device_qualifier.options.interface,
        host=server_qualifier.host,
        port=server_qualifier.port,
    )


U = TypeVar("U")


async def _gather(*args: Coroutine[Any, Any, U]) -> list[U]:
    if ASYNC_DEBUG_MODE:
        results = []
        failures = []
        for arg in args:
            try:
                # No list comprehension in debug for cleaner stack trace
                results.append(await arg)  # noqa: PERF401
            except Exception as ex:  # noqa: PERF203
                # Defer exception and continue to avoid "was never awaited" warning
                failures.append(ex)
        if len(failures) > 0:
            raise failures[0]
        return results
    return await asyncio.gather(*args)


@asynccontextmanager
async def gather_and_apply(func: Callable[[list[U]], Coroutine[Any, Any, None]]):
    awaitables: list[Coroutine[Any, Any, U]] = []
    yield awaitables
    await func(await _gather(*awaitables))


def _resolve_type(value: Any, path: str) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray) and value.dtype in [np.int32, np.int64]:
        return value.astype(dtype=np.uint32)
    if isinstance(value, np.floating):
        return float(value)
    if np.iscomplexobj(value) and "spectroscopy/envelope/wave" in path.lower():
        # TODO(2K): This conversion may not be entirely accurate, it is only known to match the expected node
        # type and address the "Vector transfer error: data have different type than expected" error.
        return preprocess_complex_shf_waveform_vector(value)[0].astype(dtype=np.uint32)
    return value


@dataclass
class AnnotatedValueWithExtras(AnnotatedValue):
    cache: bool = False
    filename: str | None = None


async def set_parallel(api: Instrument, nodes: NodeCollector):
    futures = []
    for node in nodes:
        if isinstance(node, NodeActionSet):
            func = (
                api.kernel_session.set_with_expression
                if "*" in node.path
                else api.kernel_session.set
            )
            type_adjusted_value = _resolve_type(node.value, node.path)
            if isinstance(api, MockInstrument):
                val = AnnotatedValueWithExtras(
                    path=node.path,
                    value=type_adjusted_value,
                    cache=node.cache,
                    filename=node.filename,
                )
            else:
                val = AnnotatedValue(
                    path=node.path,
                    value=type_adjusted_value,
                )
            futures.append(func(val))
        else:
            await _gather(*futures)
            futures.clear()

    if len(futures) > 0:
        await _gather(*futures)


def parse_annotated_value(annotated_value: AnnotatedValue) -> Any:
    key_translate = {"job_id": "jobid", "samples": "numsamples"}
    value = annotated_value.value
    extra_header = annotated_value.extra_header
    effective_value: Any
    if extra_header is not None:
        effective_value = [
            {
                "vector": value,
                "properties": {
                    key_translate.get(key, key): val
                    for key, val in extra_header.__dict__.items()
                },
            }
        ]
    else:
        effective_value = {"value": [value]}
    return effective_value


async def get_raw(api: Instrument, path: str) -> dict[str, Any]:
    paths = path.split(",")
    results = await _gather(*[api.kernel_session.get(p) for p in paths])
    return {r.path: parse_annotated_value(r) for r in results}


class NodeMonitorAsync(NodeMonitorBase):
    def __init__(self, api: Instrument):
        super().__init__()
        self._api = api
        self._queues: dict[str, DataQueue] = {}

    async def start(self):
        queues: list[DataQueue] = await _gather(
            *(
                self._api.kernel_session.subscribe(path, get_initial_value=True)
                for path in self._nodes.keys()
            )
        )
        for path, queue in zip(self._nodes.keys(), queues):
            self._queues[path] = queue

    async def stop(self):
        for queue in self._queues.values():
            queue.disconnect()
        self._queues.clear()
        await self.flush()

    async def poll(self):
        while True:
            # Yield to the event loop to fill queues with pending data
            await asyncio.sleep(0)
            no_more_data = True
            for path, queue in self._queues.items():
                while not queue.empty():
                    annotated_value = await queue.get()
                    self._get_node(path).append(parse_annotated_value(annotated_value))
                    no_more_data = False
            if no_more_data:
                break
