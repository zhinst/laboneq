# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum, IntFlag
import json
import logging
import shlex
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Coroutine,
    Iterator,
    Literal,
    TypeVar,
    overload,
)
from laboneq.controller.devices.device_utils import is_expected, to_l1_timeout

import numpy as np
from laboneq.controller.devices.device_utils import (
    NodeActionSet,
    NodeCollector,
    zhinst_core_version,
)

from laboneq.controller.devices.zi_emulator import EmulatorState, KernelSessionEmulator

from zhinst.comms_schemas.labone import KernelSession, KernelInfo
from zhinst.comms_schemas.labone.core import (
    DataQueue,
    AnnotatedValue,
    ShfGeneratorWaveformVectorData,
)
from zhinst.comms_schemas.labone.core.errors import NotFoundError

from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.versioning import (
    MINIMUM_SUPPORTED_LABONE_VERSION,
    RECOMMENDED_LABONE_VERSION,
    LabOneVersion,
    SetupCaps,
)

if TYPE_CHECKING:
    from laboneq.core.types.numpy_support import NumPyArray
    from laboneq.controller.devices.device_setup_dao import (
        ServerQualifier,
        DeviceQualifier,
    )

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


def parse_logfmt(log: str) -> dict[str, str]:
    # Simplified logfmt parser
    # Avoided using the logfmt library due to its apparent lack of maintenance.
    def _split_key_value(part: str) -> tuple[str, str]:
        [key, value] = part.split("=", 1)
        return key, value

    return dict(_split_key_value(part) for part in shlex.split(log) if "=" in part)


class DataServerConnection:
    _devices_node_path = "/zi/devices"
    _fullversion_node_path = "/zi/about/fullversion"
    _log_node_path = "/zi/debug/log"

    def __init__(
        self,
        server_qualifier: ServerQualifier,
        data_server: KernelSession,
        devices_json: str,
        fullversion_str: str,
        setup_caps: SetupCaps,
    ):
        self._server_qualifier = server_qualifier
        self._data_server = data_server
        self._devices_json = devices_json
        self._fullversion_str = fullversion_str
        self._log_queue: DataQueue | None = None
        self._log_records: list[AnnotatedValue] = []
        self._setup_caps = setup_caps
        self._server_setup_caps: SetupCaps | None = None

    @property
    def server_qualifier(self) -> ServerQualifier:
        return self._server_qualifier

    @property
    def setup_caps(self) -> SetupCaps:
        if self._server_setup_caps is None:
            raise LabOneQControllerException(
                "Internal error: per-server setup capabilities are not initialized."
            )
        return self._server_setup_caps

    @classmethod
    async def connect(
        cls,
        server_qualifier: ServerQualifier,
        emulator_state: EmulatorState | None,
        timeout_s: float,
        setup_caps: SetupCaps,
    ) -> DataServerConnection:
        host = server_qualifier.host
        port = server_qualifier.port
        if emulator_state is None:
            data_server = await KernelSession.create(
                kernel_info=KernelInfo.zi_connection(),
                host=host,
                port=port,
                timeout=to_l1_timeout(timeout_s),
            )
        else:
            data_server = KernelSessionEmulator(
                serial="ZI", emulator_state=emulator_state
            )  # type: ignore

        try:
            (
                devices_json,
                fullversion_str,
            ) = (
                r.value
                for r in await asyncio.gather(
                    data_server.get(cls._devices_node_path),
                    data_server.get(cls._fullversion_node_path),
                )
            )
        except NotFoundError as error:
            raise LabOneQControllerException(
                f"Data server at {host}:{port} does not provide required information. "
                "Is it supported by this version of LabOne Q?"
            ) from error

        _logger.info(
            "Connected to Zurich Instruments LabOne Data Server version %s at %s:%s",
            fullversion_str,
            host,
            port,
        )

        return DataServerConnection(
            server_qualifier=server_qualifier,
            data_server=data_server,
            devices_json=str(devices_json),
            fullversion_str=str(fullversion_str),
            setup_caps=setup_caps,
        )

    @property
    def data_server(self) -> KernelSession:
        return self._data_server

    @property
    def devices_json(self) -> str:
        return self._devices_json

    @property
    def fullversion_str(self) -> str:
        return self._fullversion_str

    def check_dataserver_device_compatibility(
        self, ignore_version_mismatch: bool, serials: list[str]
    ):
        # TODO(2K): zhinst.core version check is only relevant for the AWG compiler.
        # In the future, compile stage must store the actually used AWG compiler version
        # in the compiled experiment data, and this version has to be checked against
        # the data server version at experiment run.
        python_api_version = LabOneVersion.from_version_string(zhinst_core_version())

        dataserver_version = LabOneVersion.from_version_string(self.fullversion_str)

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
            raise LabOneQControllerException(err_msg)

        self._server_setup_caps = self._setup_caps.for_server(dataserver_version)

        if isinstance(self._data_server, KernelSession):  # Real server?
            statuses = _get_device_statuses(self.devices_json, serials)
            _check_dataserver_device_compatibility(statuses)

    async def subscribe_logs(self):
        self._log_records.clear()
        self._log_queue = await self._data_server.subscribe(
            self._log_node_path, level="debug"
        )

    async def unsubscribe_logs(self):
        if self._log_queue is not None:
            while not self._log_queue.empty():
                self._log_records.append(self._log_queue.get_nowait())
            self._log_queue.disconnect()
            self._log_queue = None

    def dump_logs(self, server_uid: str):
        logger = logging.getLogger("node.log")
        logger.debug(f"Node log from the data server with id '{server_uid}':")
        server_logger = logging.getLogger(f"server.log.{server_uid}")
        for log_record in self._log_records:
            if not isinstance(log_record.value, str):
                continue
            server_logger.debug(log_record.value)
            log_fields = parse_logfmt(log_record.value)
            tracer = log_fields.get("tracer")
            if tracer != "blocks_out":
                continue
            method = log_fields.get("method")
            path = log_fields.get("path")
            value = log_fields.get("value", "-")
            logger.debug(f"  {method} {path} {value}")


class DataServerConnections:
    def __init__(self):
        self._data_servers: dict[str, DataServerConnection | None] = {}

    def add(self, server_uid: str, data_server_connection: DataServerConnection | None):
        self._data_servers[server_uid] = data_server_connection

    def get(self, server_uid: str) -> DataServerConnection | None:
        return self._data_servers.get(server_uid)

    @property
    def _valid_data_servers(self) -> Iterator[tuple[str, DataServerConnection]]:
        for server_uid, data_server in self._data_servers.items():
            if data_server is not None:
                yield server_uid, data_server

    @asynccontextmanager
    async def capture_logs(self):
        try:
            await _gather(*(ds.subscribe_logs() for _, ds in self._valid_data_servers))
            yield
        finally:
            await _gather(
                *(ds.unsubscribe_logs() for _, ds in self._valid_data_servers)
            )
            self.dump_logs()

    def dump_logs(self):
        for server_uid, data_server in self._valid_data_servers:
            data_server.dump_logs(server_uid)


class InstrumentConnection:
    """Encapsulates the API implementation to prevent contaminating
    the rest of the controller with API dependencies."""

    def __init__(self, impl: KernelSession | None = None):
        self._impl = impl

    @staticmethod
    async def connect(
        *,
        server_qualifier: ServerQualifier,
        device_qualifier: DeviceQualifier,
        emulator_state: EmulatorState | None,
        timeout_s: float,
    ) -> InstrumentConnection:
        if emulator_state is None:
            instrument = await KernelSession.create(
                kernel_info=KernelInfo.device_connection(
                    device_id=device_qualifier.options.serial,
                    interface=device_qualifier.options.interface,
                ),
                host=server_qualifier.host,
                port=server_qualifier.port,
                timeout=to_l1_timeout(timeout_s),
            )
        else:
            instrument = KernelSessionEmulator(
                serial=device_qualifier.options.serial,
                emulator_state=emulator_state,
            )  # type: ignore
        return InstrumentConnection(instrument)

    @property
    def impl(self) -> KernelSession:
        assert self._impl is not None
        return self._impl

    def clear_cache(self):
        if isinstance(self._impl, KernelSessionEmulator):
            self._impl.clear_cache()

    async def set_parallel(self, nodes: NodeCollector):
        futures = []
        for node in nodes:
            if isinstance(node, NodeActionSet):
                func = (
                    self.impl.set_with_expression if "*" in node.path else self.impl.set
                )
                type_adjusted_value = _resolve_type(node.value, node.path)
                val: AnnotatedValue
                if isinstance(self._impl, KernelSessionEmulator):
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

    async def get_raw(self, paths: list[str]) -> dict[str, Any]:
        results = await _gather(*[self.impl.get(p) for p in paths])
        return {r.path: _from_annotated_value(r) for r in results}


U = TypeVar("U")


@overload
async def _gather(
    *args: Awaitable[U], return_exceptions: Literal[False] = False
) -> list[U]: ...


@overload
async def _gather(
    *args: Awaitable[U], return_exceptions: Literal[True]
) -> list[U | BaseException]: ...


async def _gather(
    *args: Awaitable[U], return_exceptions: bool = False
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


async def _sleep(timeout_s: float):
    await asyncio.sleep(timeout_s)


def _resolve_type(value: Any, path: str) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray) and value.dtype in [np.int32, np.int64]:
        return value.astype(dtype=np.uint32)
    if isinstance(value, np.floating):
        return float(value)
    if np.iscomplexobj(value) and (
        "generator/waveforms" in path.lower()
        or "spectroscopy/envelope/wave" in path.lower()
    ):
        return ShfGeneratorWaveformVectorData(complex=value)
    return value


@dataclass
class AnnotatedValueWithExtras(AnnotatedValue):
    cache: bool = False
    filename: str | None = None


_key_translate = {
    "job_id": "jobid",
    "jobId": "jobid",
    "samples": "numsamples",
    "first_sample_timestamp": "firstSampleTimestamp",
}


def canonical_vector(value: Any) -> NumPyArray:
    return getattr(value, "vector", value)


def canonical_properties(value: Any) -> dict[str, Any]:
    properties = getattr(value, "properties", {})
    return {_key_translate.get(key, key): val for key, val in dict(properties).items()}


def _from_annotated_value(annotated_value: AnnotatedValue) -> Any:
    value = annotated_value.value
    vector = getattr(value, "vector", None)
    if isinstance(vector, str):
        value = vector
    return value


class ResponseWaiterAsync:
    def __init__(
        self,
        api: InstrumentConnection,
        dev_repr: str,
        timeout_s: float | None = None,
    ):
        self._api = api
        self._dev_repr = dev_repr
        self._nodes: dict[str, Any] = {}
        self._messages: dict[str, str] = {}
        self._timeout_s = timeout_s
        self._queues: dict[str, DataQueue] = {}

    def add_nodes(self, nodes: dict[str, Any]):
        self._nodes.update(nodes)

    def add_with_msg(self, nodes: dict[str, tuple[Any, str]]):
        self._nodes.update({path: val[0] for path, val in nodes.items()})
        self._messages.update({path: val[1] for path, val in nodes.items()})

    async def prepare(self):
        if len(self._nodes) == 0:
            return
        queues = await _gather(
            *(self._api.impl.subscribe(path) for path in self._nodes)
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
                    else:
                        msg = f"{self._dev_repr}: {msg}"
                    failed_nodes.append(msg)
        for queue in self._queues.values():
            queue.disconnect()
        return failed_nodes


class ConditionsCheckerAsync:
    def __init__(self, api: InstrumentConnection, conditions: dict[str, Any]):
        self._api = api
        self._conditions = conditions

    async def check(self) -> list[tuple[str, Any, Any]]:
        results: list[AnnotatedValue] = await _gather(
            *(self._api.impl.get(path) for path in self._conditions)
        )
        values = [(res.path, _from_annotated_value(res)) for res in results]
        mismatches = [
            (path, value, self._conditions[path])
            for path, value in values
            if not is_expected(value, self._conditions[path])
        ]
        return mismatches

    async def wait_by_get(self):
        while len(await self.check()) > 0:
            pass


@dataclass
class ResultData:
    vector: NumPyArray
    properties: dict[str, Any]


class AsyncSubscriber:
    def __init__(self):
        self._subscriptions: dict[str, DataQueue] = {}
        self._last: dict[str, AnnotatedValue] = {}

    async def subscribe(self, api: InstrumentConnection, path: str, get_initial=False):
        if path in self._subscriptions:
            return  # Keep existing subscription
        self._subscriptions[path] = await api.impl.subscribe(path)
        if get_initial:
            self._last[path] = await api.impl.get(path)

    async def get(self, path: str, timeout_s: float | None = None) -> AnnotatedValue:
        queue = self._subscriptions.get(path)
        assert queue is not None, f"path {path} is not subscribed"
        value = await asyncio.wait_for(queue.get(), timeout=timeout_s)
        self._last[path] = value
        return value

    async def get_result(self, path: str, timeout_s: float | None = None) -> ResultData:
        value = await self.get(path=path, timeout_s=timeout_s)
        return ResultData(
            vector=canonical_vector(value.value),
            properties=canonical_properties(value.value),
        )

    def get_updates(self, path) -> list[AnnotatedValue]:
        updates = []
        queue = self._subscriptions.get(path)
        if queue is not None:
            while not queue.empty():
                value = queue.get_nowait()
                self._last[path] = value
                updates.append(value)
        return updates

    def unsubscribe_all(self):
        for queue in self._subscriptions.values():
            queue.disconnect()
        self._subscriptions.clear()
        self._last.clear()

    async def wait_for(self, path: str, expected: Any, timeout_s: float):
        self.get_updates(path)
        while path not in self._last or not is_expected(
            _from_annotated_value(self._last[path]), expected
        ):
            await self.get(path, timeout_s=timeout_s)
