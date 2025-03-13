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
    return dict(tuple(part.split("=", 1)) for part in shlex.split(log) if "=" in part)


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
    ):
        self._server_qualifier = server_qualifier
        self._data_server = data_server
        self._devices_json = devices_json
        self._fullversion_str = fullversion_str
        self._log_queue: DataQueue | None = None
        self._log_records: list[AnnotatedValue] = []

    @property
    def server_qualifier(self) -> ServerQualifier:
        return self._server_qualifier

    @classmethod
    async def connect(
        cls,
        server_qualifier: ServerQualifier,
        emulator_state: EmulatorState | None,
        timeout_s: float,
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
            )

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
            server_qualifier, data_server, str(devices_json), str(fullversion_str)
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
        for log_record in self._log_records:
            if not isinstance(log_record.value, str):
                continue
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


async def create_device_kernel_session(
    *,
    server_qualifier: ServerQualifier,
    device_qualifier: DeviceQualifier,
    emulator_state: EmulatorState | None,
    timeout_s: float,
) -> KernelSession:
    if emulator_state is not None:
        return KernelSessionEmulator(
            serial=device_qualifier.options.serial,
            emulator_state=emulator_state,
        )
    return await KernelSession.create(
        kernel_info=KernelInfo.device_connection(
            device_id=device_qualifier.options.serial,
            interface=device_qualifier.options.interface,
        ),
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


def _preprocess_complex_shf_waveform_vector(
    data: np.ndarray,
) -> np.ndarray:
    """Preprocess complex waveform vector data.

    Complex waveform vectors are transmitted as two uint32 interleaved vectors.
    This function converts the complex waveform vector data into the
    corresponding uint32 vector.

    Args:
        data: The complex waveform vector data.

    Returns:
        The uint32 vector data.
    """
    _SHF_WAVEFORM_UNSIGNED_ENCODING_BITS = 18
    _SHF_WAVEFORM_SIGNED_ENCODING_BITS = _SHF_WAVEFORM_UNSIGNED_ENCODING_BITS - 1
    _SHF_WAVEFORM_SCALING = (1 << _SHF_WAVEFORM_SIGNED_ENCODING_BITS) - 1
    real_scaled = np.round(np.real(data) * _SHF_WAVEFORM_SCALING).astype(np.uint32)
    imag_scaled = np.round(np.imag(data) * _SHF_WAVEFORM_SCALING).astype(np.uint32)
    decoded_data = np.empty((2 * data.size,), dtype=np.uint32)
    decoded_data[::2] = real_scaled
    decoded_data[1::2] = imag_scaled
    return decoded_data


def _resolve_type(value: Any, path: str) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray) and value.dtype in [np.int32, np.int64]:
        return value.astype(dtype=np.uint32)
    if isinstance(value, np.floating):
        return float(value)
    if np.iscomplexobj(value) and "spectroscopy/envelope/wave" in path.lower():
        # TODO(2K): This conversion address the "Vector transfer error: data have different type than expected"
        # error. API should accept complex vector for the node instead, as per
        # https://docs.zhinst.com/shfqa_user_manual/nodedoc.html#qachannels.
        return _preprocess_complex_shf_waveform_vector(value)
    return value


@dataclass
class AnnotatedValueWithExtras(AnnotatedValue):
    cache: bool = False
    filename: str | None = None


async def set_parallel(api: KernelSession, nodes: NodeCollector):
    futures = []
    for node in nodes:
        if isinstance(node, NodeActionSet):
            func = api.set_with_expression if "*" in node.path else api.set
            type_adjusted_value = _resolve_type(node.value, node.path)
            if isinstance(api, KernelSessionEmulator):
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


async def get_raw(api: KernelSession, path: str) -> dict[str, Any]:
    paths = path.split(",")
    results = await _gather(*[api.get(p) for p in paths])
    return {r.path: parse_annotated_value(r) for r in results}


async def _yield():
    # Yield to the event loop to fill queues with pending data
    # Note: asyncio.sleep(0) is not sufficient. See:
    # https://stackoverflow.com/questions/74493571/asyncio-sleep0-does-not-yield-control-to-the-event-loop
    # https://bugs.python.org/issue40800
    # TODO(2K): rework the logic to use proper async once the legacy API is removed.
    await asyncio.sleep(0.0001)


async def wait_for_state_change(
    api: KernelSession,
    path: str,
    value: int | str,
):
    queue = await api.subscribe(path, get_initial_value=True)
    node_value = _from_annotated_value(await queue.get())
    while value != node_value:
        node_value = _from_annotated_value(await queue.get())
    queue.disconnect()


def _from_annotated_value(annotated_value: AnnotatedValue) -> Any:
    value = annotated_value.value
    vector = getattr(value, "vector", None)
    if isinstance(vector, str):
        value = vector
    return value


class ResponseWaiterAsync:
    def __init__(
        self,
        api: KernelSession,
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

    async def prepare(self):
        if len(self._nodes) == 0:
            return
        queues = await _gather(*(self._api.subscribe(path) for path in self._nodes))
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
    def __init__(self, api: KernelSession, conditions: dict[str, Any]):
        self._api = api
        self._conditions = conditions

    async def check(self) -> list[tuple[str, Any, Any]]:
        results: list[AnnotatedValue] = await _gather(
            *(self._api.get(path) for path in self._conditions)
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
        self, api: KernelSession, path: str, get_initial_value: bool = False
    ):
        if path in self._subscriptions:
            if get_initial_value:
                self._subscriptions.pop(path).disconnect()
            else:
                return  # Keep existing subscription
        self._subscriptions[path] = await api.subscribe(
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
