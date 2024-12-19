# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum, IntFlag
import json
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Literal, TypeVar, overload
from laboneq.controller.devices.device_utils import is_expected, to_l1_timeout

import numpy as np
from laboneq.controller.devices.device_utils import (
    NodeActionSet,
    NodeCollector,
    zhinst_core_version,
)

from laboneq.controller.devices.zi_emulator import EmulatorState, MockInstrument
from laboneq.controller.devices.zi_node_monitor import NodeMonitorBase

from labone import DataServer, Instrument
from labone.core import AnnotatedValue
from labone.core.subscription import DataQueue
from labone.core.shf_vector_data import (
    ShfGeneratorWaveformVectorData,
    preprocess_complex_shf_waveform_vector,
)

from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.versioning import (
    MINIMUM_SUPPORTED_LABONE_VERSION,
    RECOMMENDED_LABONE_VERSION,
    LabOneVersion,
)

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.controller.communication import ServerQualifier
    from laboneq.controller.devices.device_zi import DeviceQualifier

_logger = logging.getLogger(__name__)


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


async def _get_data_server_info(dataserver: DataServer) -> tuple[str, str, int]:
    devices_node_path = "/zi/devices"
    version_node_path = "/zi/about/version"
    revision_node_path = "/zi/about/revision"

    devices_json, version_str, revision_int = (
        r.value
        for r in await asyncio.gather(
            dataserver.kernel_session.get(devices_node_path),
            dataserver.kernel_session.get(version_node_path),
            dataserver.kernel_session.get(revision_node_path),
        )
    )

    return devices_json, version_str, revision_int


def _get_device_statuses(
    devices_json: str, serials: list[str]
) -> dict[str, _DeviceStatusFlag]:
    devices = json.loads(devices_json)
    try:
        return {
            serial: _DeviceStatusFlag(devices[serial.upper()]["STATUSFLAGS"])
            for serial in serials
        }
    except KeyError as error:
        raise LabOneQControllerException(
            f"Device {error} could not be found."
        ) from error


def _check_dataserver_device_compatibility(statuses: dict[str, _DeviceStatusFlag]):
    errors = []
    for serial, flags in statuses.items():
        if _DeviceStatusFlag.FW_UPDATE_IN_PROGRESS in flags:
            raise LabOneQControllerException(
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
        raise LabOneQControllerException(
            "LabOne and device firmware version compatibility issues were found.\n"
            + "\n".join(errors)
        )


async def async_check_dataserver_device_compatibility(
    host: str,
    port: int,
    serials: list[str],
    emulator_state: EmulatorState | None,
    ignore_version_mismatch: bool,
    timeout_s: float,
):
    if port == -1:  # Dummy server
        return

    # TODO(2K): zhinst.core version check is only relevant for the AWG compiler.
    # In the future, compile stage must store the actually used AWG compiler version
    # in the compiled experiment data, and this version has to be checked against
    # the data server version at experiment run.
    python_api_version = LabOneVersion.from_version_string(zhinst_core_version())

    if emulator_state is None:
        dataserver = await DataServer.create(
            host=host, port=port, timeout=to_l1_timeout(timeout_s)
        )
    else:
        dataserver = MockInstrument(serial="ZI", emulator_state=emulator_state)

    devices_json, version_str, revision_int = await _get_data_server_info(dataserver)
    dataserver_version = LabOneVersion.from_dataserver_version_information(
        version=version_str, revision=revision_int
    )

    if dataserver_version != python_api_version:
        err_msg = f"Version of LabOne Data Server ({dataserver_version}) and Python API ({python_api_version}) do not match."
        if ignore_version_mismatch:
            _logger.warning("Ignoring that %s", err_msg)
        else:
            raise LabOneQControllerException(err_msg)
    elif dataserver_version < MINIMUM_SUPPORTED_LABONE_VERSION:
        err_msg = (
            f"Version of LabOne Data Server '{dataserver_version}' is not supported. "
            f"We recommend {RECOMMENDED_LABONE_VERSION}."
        )
        if ignore_version_mismatch:
            _logger.warning("Ignoring that %s", err_msg)
        else:
            raise LabOneQControllerException(err_msg)

    _logger.info(
        "Connected to Zurich Instruments LabOne Data Server version %s at %s:%s",
        version_str,
        host,
        port,
    )

    if emulator_state is None:  # real server
        statuses = _get_device_statuses(devices_json, serials)
        _check_dataserver_device_compatibility(statuses)


async def create_device_kernel_session(
    *,
    server_qualifier: ServerQualifier,
    device_qualifier: DeviceQualifier,
    emulator_state: EmulatorState | None,
    timeout_s: float,
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
        timeout=to_l1_timeout(timeout_s),
    )


U = TypeVar("U")


@overload
async def _gather(
    *args: Coroutine[Any, Any, U], return_exceptions: Literal[False] = False
) -> list[U]: ...


@overload
async def _gather(
    *args: Coroutine[Any, Any, U], return_exceptions: Literal[True]
) -> list[U | BaseException]: ...


async def _gather(
    *args: Coroutine[Any, Any, U], return_exceptions: bool = False
) -> list[U] | list[U | BaseException]:
    if ASYNC_DEBUG_MODE:
        results: list[U | BaseException] = []
        failures = []
        for arg in args:
            try:
                # No list comprehension in debug for cleaner stack trace
                results.append(await arg)  # noqa: PERF401
            except Exception as ex:  # noqa: PERF203
                # Defer exception and continue to avoid "was never awaited" warning
                if return_exceptions:
                    results.append(ex)
                else:
                    failures.append(ex)
        if len(failures) > 0:
            raise failures[0]
        return results
    return await asyncio.gather(*args, return_exceptions=return_exceptions)


async def _gather_with_timeout(
    *args: Coroutine[Any, Any, U], timeout_s: float
) -> list[U | BaseException]:
    return await _gather(
        *(asyncio.wait_for(arg, timeout=timeout_s) for arg in args),
        return_exceptions=True,
    )


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
        return np.frombuffer(
            preprocess_complex_shf_waveform_vector(value)["vectorData"]["data"],
            dtype=np.uint32,
        )
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
                # Don't wrap into ShfGeneratorWaveformVectorData for emulation
                # (see also code in "else" below)
                # to avoid emulator dependency on labone-python
                val = AnnotatedValueWithExtras(
                    path=node.path,
                    value=type_adjusted_value,
                    cache=node.cache,
                    filename=node.filename,
                )
            else:
                if (
                    np.iscomplexobj(type_adjusted_value)
                    and "generator/waveforms" in node.path.lower()
                ):
                    type_adjusted_value = ShfGeneratorWaveformVectorData(
                        complex=type_adjusted_value
                    )
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


_key_translate = {
    "job_id": "jobid",
    "jobId": "jobid",
    "samples": "numsamples",
    "first_sample_timestamp": "firstSampleTimestamp",
}


def canonical_vector(value: Any) -> NumPyArray:
    return getattr(value, "vector", value)


def canonical_properties(properties: Any) -> dict[str, Any]:
    return {_key_translate.get(key, key): val for key, val in dict(properties).items()}


def parse_annotated_value(annotated_value: AnnotatedValue) -> Any:
    value = annotated_value.value
    properties = getattr(value, "properties", None)
    effective_value: Any
    if properties is not None:
        effective_value = [
            {
                "vector": value.vector,
                "properties": canonical_properties(properties),
            }
        ]
    else:
        if isinstance(value, np.ndarray):
            effective_value = [{"vector": value}]
        else:
            effective_value = {"value": [value]}
    return effective_value


async def get_raw(api: Instrument, path: str) -> dict[str, Any]:
    paths = path.split(",")
    results = await _gather(*[api.kernel_session.get(p) for p in paths])
    return {r.path: parse_annotated_value(r) for r in results}


async def _yield():
    # Yield to the event loop to fill queues with pending data
    # Note: asyncio.sleep(0) is not sufficient. See:
    # https://stackoverflow.com/questions/74493571/asyncio-sleep0-does-not-yield-control-to-the-event-loop
    # https://bugs.python.org/issue40800
    # TODO(2K): rework the logic to use proper async once the legacy API is removed.
    await asyncio.sleep(0.0001)


async def wait_for_state_change(
    api: Instrument,
    path: str,
    value: int | str,
):
    queue = await api.kernel_session.subscribe(path, get_initial_value=True)
    node_value = _from_annotated_value(await queue.get())
    while value != node_value:
        node_value = _from_annotated_value(await queue.get())
    queue.disconnect()


class NodeMonitorAsync(NodeMonitorBase):
    def __init__(self, api: Instrument):
        super().__init__()

    async def start(self):
        pass

    async def stop(self):
        pass

    async def poll(self):
        pass

    async def wait_for_state_by_get(self, path: str, expected: int):
        pass


def _from_annotated_value(annotated_value: AnnotatedValue) -> Any:
    value = annotated_value.value
    vector = getattr(value, "vector", None)
    if isinstance(vector, str):
        value = vector
    return value


class ResponseWaiterAsync:
    def __init__(
        self,
        api: Instrument,
        nodes: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ):
        self._api = api
        self._nodes: dict[str, Any] = {}
        self._messages: dict[str, str] = {}
        self._timeout_s = timeout_s
        self._queues: dict[str, DataQueue] = {}
        if nodes is not None:
            self.add_nodes(nodes)

    def add_nodes(self, nodes: dict[str, Any]):
        self._nodes.update(nodes)

    def add_with_msg(self, nodes: dict[str, tuple[Any, str]]):
        self._nodes.update({path: val[0] for path, val in nodes.items()})
        self._messages.update({path: val[1] for path, val in nodes.items()})

    async def prepare(self, get_initial_value: bool = False):
        if len(self._nodes) == 0:
            return
        queues = await _gather(
            *(
                self._api.kernel_session.subscribe(
                    path, get_initial_value=get_initial_value
                )
                for path in self._nodes
            )
        )
        for path, queue in zip(self._nodes.keys(), queues):
            self._queues[path] = queue

    async def _wait_one(self, path: str):
        queue = self._queues[path]
        expected = self._nodes[path]
        all_expected = []
        if isinstance(expected, list):
            all_expected.extend(expected)
        else:
            all_expected.append(expected)
        while len(all_expected) > 0:
            node_value = _from_annotated_value(await queue.get())
            if is_expected(node_value, all_expected[0]):
                all_expected.pop(0)

    async def wait(self) -> list[str]:
        if len(self._nodes) == 0:
            return []
        failed_nodes: list[str] = []
        if self._timeout_s is None:
            await _gather(*(self._wait_one(path) for path in self._nodes))
        else:
            results = await _gather_with_timeout(
                *(self._wait_one(path) for path in self._nodes),
                timeout_s=self._timeout_s,
            )
            for result, (path, expected) in zip(results, self._nodes.items()):
                if isinstance(result, (asyncio.TimeoutError, TimeoutError)):
                    msg = self._messages.get(path)
                    if msg is None:
                        msg = f"{path}={expected}"
                    failed_nodes.append(msg)
        for queue in self._queues.values():
            queue.disconnect()
        return failed_nodes


class ConditionsCheckerAsync:
    def __init__(self, api: Instrument, conditions: dict[str, Any]):
        self._api = api
        self._conditions = conditions

    async def check(self) -> list[tuple[str, Any, Any]]:
        results: list[AnnotatedValue] = await _gather(
            *(self._api.kernel_session.get(path) for path in self._conditions)
        )
        values = [(res.path, _from_annotated_value(res)) for res in results]
        mismatches = [
            (path, value, self._conditions[path])
            for path, value in values
            if not is_expected(value, self._conditions[path])
        ]
        return mismatches


class AsyncSubscriber:
    def __init__(self):
        self._subscriptions: dict[str, DataQueue] = {}

    async def subscribe(
        self, api: Instrument, path: str, get_initial_value: bool = False
    ):
        if path in self._subscriptions:
            if get_initial_value:
                self._subscriptions.pop(path).disconnect()
            else:
                return  # Keep existing subscription
        self._subscriptions[path] = await api.kernel_session.subscribe(
            path, get_initial_value=get_initial_value
        )

    async def get(self, path: str, timeout_s: float | None = None) -> AnnotatedValue:
        queue = self._subscriptions.get(path)
        assert queue is not None, f"path {path} is not subscribed"
        return await asyncio.wait_for(queue.get(), timeout=timeout_s)

    def get_updates(self, path) -> list[AnnotatedValue]:
        updates = []
        queue = self._subscriptions.get(path)
        if queue is not None:
            while not queue.empty():
                updates.append(queue.get_nowait())
        return updates

    def unsubscribe_all(self):
        for queue in self._subscriptions.values():
            queue.disconnect()
        self._subscriptions.clear()
