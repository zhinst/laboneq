# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from contextlib import asynccontextmanager
from enum import Enum
import json

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Iterator, cast

from laboneq.controller.devices.async_support import (
    ConditionsCheckerAsync,
    DataServerConnection,
    DataServerConnections,
    _gather,
    _gather_with_timeout,
    wait_for_state_change,
)
from laboneq.controller.devices.device_factory import DeviceFactory
from laboneq.controller.devices.device_qhub import DeviceQHUB
from laboneq.controller.devices.device_setup_dao import DeviceSetupDAO, ServerQualifier
from laboneq.controller.devices.device_utils import (
    NodeCollector,
    prepare_emulator_state,
)
from laboneq.controller.devices.device_zi import DeviceBase, DeviceZI
from laboneq.controller.devices.zi_emulator import EmulatorState
from laboneq.controller.devices.node_control import (
    NodeControlBase,
    NodeControlKind,
    filter_states,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.controller.versioning import SetupCaps
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource

if TYPE_CHECKING:
    from laboneq.data.execution_payload import TargetSetup


SUPPRESSED_WARNINGS = frozenset(
    {
        "AWGDIOTIMING",  # SVT-120, has no consequence for LabOne Q usage of DIO.
    }
)


_logger = logging.getLogger(__name__)


DEFAULT_TIMEOUT_S = 10


class DeviceCollection:
    def __init__(
        self,
        target_setup: TargetSetup,
        setup_caps: SetupCaps,
        ignore_version_mismatch: bool = False,
    ):
        self._ds = DeviceSetupDAO(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
            setup_caps=setup_caps,
        )
        if self._ds.has_uhf and self._ds.has_qhub:
            raise LabOneQControllerException("Gen1 setup with QHub is not supported.")
        self._emulator_state: EmulatorState | None = None
        self._ignore_version_mismatch = ignore_version_mismatch
        self._timeout_s: float = DEFAULT_TIMEOUT_S
        self._data_servers = DataServerConnections()
        self._devices: dict[str, DeviceZI] = {}

    @property
    def emulator_state(self) -> EmulatorState:
        if self._emulator_state is None:
            self._emulator_state = prepare_emulator_state(self._ds)
        return self._emulator_state

    @property
    def devices(self) -> dict[str, DeviceZI]:
        return self._devices

    @property
    def all(self) -> Iterator[tuple[str, DeviceZI]]:
        for uid, device in self._devices.items():
            yield uid, device

    @property
    def has_qhub(self) -> bool:
        return self._ds.has_qhub

    @asynccontextmanager
    async def capture_logs(self):
        async with self._data_servers.capture_logs():
            yield

    def set_timeout(self, timeout_s: float | None):
        self._timeout_s = DEFAULT_TIMEOUT_S if timeout_s is None else timeout_s

    def find_by_uid(self, device_uid) -> DeviceZI:
        device = self._devices.get(device_uid)
        if device is None:
            raise LabOneQControllerException(
                f"Could not find device object for the device uid '{device_uid}'"
            )
        return device

    def find_by_node_path(self, path: str) -> DeviceZI:
        if m := re.match(r"^/?(dev[^/]+)/.+", path.lower()):
            serial = m.group(1)
            for _, dev in self.all:
                if dev.serial == serial:
                    return dev
        raise LabOneQControllerException(f"Could not find device for the path '{path}'")

    async def connect(
        self,
        do_emulation: bool,
        reset_devices: bool = False,
        disable_runtime_checks: bool = False,
    ):
        await self._prepare_daqs(do_emulation=do_emulation)
        await self._validate_dataserver_device_fw_compatibility()
        self._prepare_devices()
        for _, device in self.all:
            await device.connect(
                self.emulator_state if do_emulation else None,
                disable_runtime_checks=disable_runtime_checks,
                timeout_s=self._timeout_s,
            )
        async with self.capture_logs():
            await self.configure_device_setup(reset_devices=reset_devices)

    async def _configure_async(
        self,
        devices: list[DeviceZI],
        control_nodes_getter: Callable[[DeviceZI], list[NodeControlBase]],
        config_name: str,
    ):
        await _gather(
            *(
                device.exec_config_step(
                    control_nodes=control_nodes_getter(device),
                    config_name=config_name,
                    timeout_s=self._timeout_s,
                )
                for device in devices
            )
        )

    async def configure_device_setup(self, reset_devices: bool):
        _logger.info("Configuring the device setup")

        def _followers_of(parents: list[DeviceZI]) -> list[DeviceZI]:
            children = []
            for parent_dev in parents:
                for down_stream_devices in parent_dev._downlinks.values():
                    for _, dev_ref in down_stream_devices:
                        if (dev := dev_ref()) is not None:
                            children.append(dev)
            return children

        if reset_devices:
            await self._configure_async(
                [device for _, device in self.all if not device.is_secondary],
                lambda d: cast(DeviceZI, d).load_factory_preset_control_nodes(),
                "Reset to factory defaults",
            )
            # TODO(2K): Error check
            for _, device in self.all:
                device.clear_cache()

        if self.has_qhub:
            leaders = [dev for dev in self._devices.values() if dev.is_leader()]
            followers = _followers_of(leaders)
            # Switch QHUB to the external clock as usual
            await self._configure_async(
                leaders,
                lambda d: cast(DeviceZI, d).clock_source_control_nodes(),
                "QHUB: Reference clock switching",
            )
            # Check if zsync link is already established
            async_checkers: list[ConditionsCheckerAsync] = []
            for dev in followers:
                if isinstance(dev, DeviceBase):
                    async_checker = ConditionsCheckerAsync(
                        dev._api,
                        {
                            n.path: n.value
                            for n in filter_states(dev.clock_source_control_nodes())
                        },
                    )
                async_checkers.append(async_checker)
            for dev in leaders:
                if isinstance(dev, DeviceBase):
                    async_checker = ConditionsCheckerAsync(
                        dev._api,
                        {
                            n.path: n.value
                            for n in filter_states(dev.zsync_link_control_nodes())
                        },
                    )
                async_checkers.append(async_checker)
            dev_results = await _gather(
                *(async_checker.check() for async_checker in async_checkers)
            )
            failed = [res for dev_res in dev_results for res in dev_res]
            if len(failed) > 0:
                # Switch followers to ZSync, but don't wait for completion yet
                for dev in followers:
                    ref_set = next(
                        (
                            c
                            for c in dev.clock_source_control_nodes()
                            if c.kind == NodeControlKind.Setting
                        ),
                        None,
                    )
                    if ref_set is not None:
                        nc = NodeCollector()
                        nc.add(ref_set.path, ref_set.value)
                        await dev.set_async(nc)
                # Reset the QHub phy. TODO(2K): This is the actual workaround, remove this whole block once fixed.
                for dev in leaders:
                    if isinstance(dev, DeviceQHUB):
                        await dev.qhub_reset_zsync_phy()
                # Wait for completion using the dumb get method, as the streamed state sequence may be unreliable
                # with this workaround due to unexpected state bouncing.
                results = await _gather_with_timeout(
                    *(
                        wait_for_state_change(async_checker._api, path, expected)
                        for async_checker in async_checkers
                        for path, expected in async_checker._conditions.items()
                    ),
                    timeout_s=10,  # TODO(2K): use timeout passed to connect
                )
                for res in results:
                    if isinstance(res, Exception):
                        raise res
            # From here we will continue with the regular config. Since the ZSync link is now established,
            # the respective actions from the regular config will be skipped.

        configs = {
            "Configure runtime checks": lambda d: cast(
                DeviceZI, d
            ).runtime_check_control_nodes(),
            "Reference clock switching": lambda d: cast(
                DeviceZI, d
            ).clock_source_control_nodes(),
            "System frequency switching": lambda d: cast(
                DeviceZI, d
            ).system_freq_control_nodes(),
            "Setting RF channel offsets": lambda d: cast(
                DeviceZI, d
            ).rf_offset_control_nodes(),
            "Establishing ZSync link": lambda d: cast(
                DeviceZI, d
            ).zsync_link_control_nodes(),
        }

        leaders = [
            dev
            for dev in self._devices.values()
            if dev.is_leader() or dev.is_standalone()
        ]
        for config_name, control_nodes_getter in configs.items():
            # Begin by applying a config step from the leader(s), and then proceed with
            # the downstream devices. This is because downstream devices may rely on the
            # settings of upstream devices, such as an external reference supply. If there
            # are any dependencies in the reverse direction, such as the status of a ZSync
            # link, they should be resolved by moving the dependent settings to a later
            # configuration step.
            targets = leaders
            while len(targets) > 0:
                await self._configure_async(targets, control_nodes_getter, config_name)
                targets = _followers_of(targets)
        _logger.info("The device setup is configured")

    async def disconnect(self):
        for _, device in self.all:
            device.disconnect()
        self._devices = {}
        self._data_servers = DataServerConnections()
        self._emulator_state = None

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        # Set of outputs to disable or skip (depending on the 'invert' param) per device.
        # Rationale for the logic: the actual number of outputs is only known by the connected
        # device object, here we can only determine the outputs mapped in the device setup.
        outputs_per_device: dict[str, set[int]] = defaultdict(set)

        if logical_signals is None:
            invert = True
            known_device_uids = [uid for uid, _ in self.all]
            if device_uids is None:
                device_uids = known_device_uids
            else:
                device_uids = [uid for uid in device_uids if uid in known_device_uids]
            for device_uid in device_uids:
                outputs_per_device[device_uid] = (
                    self._ds.get_device_used_outputs(device_uid)
                    if unused_only
                    else set()
                )
        else:
            invert = False
            assert device_uids is None and not unused_only
            for ls_path in logical_signals:
                maybe_device_uid, outputs = self._ds.resolve_ls_path_outputs(ls_path)
                if maybe_device_uid is not None:
                    outputs_per_device[maybe_device_uid].update(outputs)

        await _gather(
            *(
                self.find_by_uid(device_uid).disable_outputs(outputs, invert)
                for device_uid, outputs in outputs_per_device.items()
            )
        )

    def for_each_sync(
        self,
        device_method: Callable[..., Any],
        *args,
        **kwargs,
    ) -> list[Any]:
        """Call a method on all devices of a given class."""
        [class_name, method_name] = device_method.__qualname__.split(".", 1)
        device_class = device_method.__globals__[class_name]
        return [
            # To keep polymorph behavior, we use getattr on the object instance
            # to retrieve the actual method to call, using passed method only as
            # a reference to the method name.
            getattr(device, method_name)(*args, **kwargs)
            for device in self._devices.values()
            if isinstance(device, device_class)
        ]

    async def for_each(
        self,
        device_method: Callable[..., Coroutine[Any, Any, None]],
        *args,
        **kwargs,
    ):
        """Call an async method on all devices of a given class in parallel."""
        await _gather(*self.for_each_sync(device_method, *args, **kwargs))

    async def _validate_dataserver_device_fw_compatibility(self):
        """Validate dataserver and device firmware compatibility."""

        daq_dev_serials: dict[str, list[str]] = defaultdict(list)
        for device_qualifier in self._ds.instruments:
            daq_dev_serials[device_qualifier.server_uid].append(
                device_qualifier.options.serial
            )

        for server_uid, dev_serials in daq_dev_serials.items():
            data_server_connection = self._data_servers.get(server_uid)
            if data_server_connection is not None:
                data_server_connection.check_dataserver_device_compatibility(
                    self._ignore_version_mismatch,
                    dev_serials,
                )

    def _prepare_devices(self):
        updated_devices: dict[str, DeviceZI] = {}
        for device_qualifier in self._ds.instruments:
            device = self._devices.get(device_qualifier.uid)
            if device is None or device.device_qualifier != device_qualifier:
                data_server = self._data_servers.get(device_qualifier.server_uid)
                server_qualifier = (
                    ServerQualifier()
                    if data_server is None
                    else data_server.server_qualifier
                )
                device = DeviceFactory.create(
                    server_qualifier, device_qualifier, self._ds.setup_caps
                )
            device.remove_all_links()
            updated_devices[device_qualifier.uid] = device
        self._devices = updated_devices

        # Update device links and leader/follower status
        for device_qualifier in self._ds.instruments:
            from_dev = self._devices[device_qualifier.uid]
            for from_port, to_dev_uid in self._ds.downlinks_by_device_uid(
                device_qualifier.uid
            ):
                to_dev = self._devices.get(to_dev_uid)
                if to_dev is None:
                    raise LabOneQControllerException(
                        f"Could not find destination device '{to_dev_uid}' for "
                        f"the port '{from_port}' of the device '{device_qualifier.uid}'"
                    )
                from_dev.add_downlink(from_port, to_dev_uid, to_dev)
                to_dev.add_uplink(from_dev)

        # Move various device settings from device setup
        for device_qualifier in self._ds.instruments:
            dev = self._devices[device_qualifier.uid]

            # Set the clock source (external by default)
            # TODO(2K): Simplify the logic in this code snippet and the one in 'update_clock_source'.
            # Currently, it adheres to the previously existing logic in the compiler, but it appears
            # unnecessarily convoluted.
            force_internal: bool | None = None
            if device_qualifier.options.reference_clock_source is not None:
                force_internal = (
                    device_qualifier.options.reference_clock_source
                    == ReferenceClockSource.INTERNAL
                )
            dev.update_clock_source(force_internal)

            dev.update_from_device_setup(self._ds)

    async def _prepare_daqs(self, do_emulation: bool):
        updated_data_servers = DataServerConnections()
        for server_uid, server_qualifier in self._ds.servers:
            existing = updated_data_servers.get(server_uid)
            if existing is not None and existing.server_qualifier == server_qualifier:
                updated_data_servers.add(server_uid, self._data_servers.get(server_uid))
                continue

            if server_qualifier.port == -1:  # Dummy server
                data_server_connection = None
            else:
                _logger.info(
                    "Connecting to data server at %s:%s",
                    server_qualifier.host,
                    server_qualifier.port,
                )
                data_server_connection = await DataServerConnection.connect(
                    server_qualifier,
                    self.emulator_state if do_emulation else None,
                    self._timeout_s,
                )
            updated_data_servers.add(server_uid, data_server_connection)
        self._data_servers = updated_data_servers

    async def fetch_device_errors(self) -> str | None:
        all_errors: list[str] = []
        for _, device in self.all:
            dev_errors = await device.fetch_errors()
            all_errors.extend(decode_errors(dev_errors, device.dev_repr))
        if len(all_errors) == 0:
            return None
        all_messages = "\n".join(all_errors)
        return f"Error(s) happened on device(s) during the execution of the experiment. Error messages:\n{all_messages}"


class DeviceErrorSeverity(Enum):
    info = 0
    warning = 1
    error = 2


def decode_errors(errors: str | list[str], dev_repr: str) -> list[str]:
    collected_messages: list[str] = []
    if not isinstance(errors, list):
        errors = [errors]
    for error in errors:
        # the key in error["vector"] looks like a dict, but it's a string. so we have to use
        # json.loads to convert it into a dict.
        val = error.get("vector")
        if val is None:
            val = error.get("value")[-1]
        error_vector = json.loads(val)
        for message in error_vector["messages"]:
            severity = DeviceErrorSeverity(message["severity"])
            if message["code"] == "AWGRUNTIMEERROR" and message["params"][0] == 1:
                awg_core = int(message["attribs"][0])
                program_counter = int(message["params"][1])
                collected_messages.append(
                    f"{dev_repr}: Gap detected on AWG core {awg_core}, program counter {program_counter}"
                )
            if severity in [
                DeviceErrorSeverity.warning,
                DeviceErrorSeverity.error,
            ] and (message["code"] not in SUPPRESSED_WARNINGS):
                collected_messages.append(
                    f"{dev_repr}: ({message['code']}) {message['message']}"
                )
    return collected_messages
