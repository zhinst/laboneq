# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from enum import Enum
import json

import logging
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Iterator, cast

from zhinst.utils.api_compatibility import check_dataserver_device_compatibility

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeSetAction,
    DaqWrapper,
    DaqWrapperDryRun,
    batch_set,
)
from laboneq.controller.devices.device_factory import DeviceFactory
from laboneq.controller.devices.device_setup_dao import DeviceSetupDAO
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.devices.zi_node_monitor import (
    ConditionsChecker,
    NodeControlBase,
    NodeControlKind,
    ResponseWaiter,
    filter_commands,
    filter_conditions,
    filter_responses,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource

if TYPE_CHECKING:
    from laboneq.data.execution_payload import TargetSetup


_logger = logging.getLogger(__name__)


class DeviceCollection:
    def __init__(
        self,
        target_setup: TargetSetup,
        dry_run: bool,
        ignore_version_mismatch: bool = False,
        reset_devices: bool = False,
    ):
        self._ds = DeviceSetupDAO(
            target_setup=target_setup,
            dry_run=dry_run,
            ignore_version_mismatch=ignore_version_mismatch,
        )
        self._dry_run = dry_run
        self._ignore_version_mismatch = ignore_version_mismatch
        self._reset_devices = reset_devices
        self._daqs: dict[str, DaqWrapper] = {}
        self._devices: dict[str, DeviceZI] = {}
        self._monitor_started = False

    @property
    def all(self) -> Iterator[tuple[str, DeviceZI]]:
        for uid, device in self._devices.items():
            yield uid, device

    @property
    def leaders(self) -> Iterator[tuple[str, DeviceZI]]:
        for uid, device in self._devices.items():
            if device.is_leader():
                yield uid, device

    @property
    def followers(self) -> Iterator[tuple[str, DeviceZI]]:
        for uid, device in self._devices.items():
            if device.is_follower():
                yield uid, device

    def find_by_uid(self, device_uid) -> DeviceZI:
        device = self._devices.get(device_uid)
        if device is None:
            raise LabOneQControllerException(
                f"Could not find device object for the device uid '{device_uid}'"
            )
        return device

    def find_by_node_path(self, path: str) -> DeviceZI:
        m = re.match(r"^/?(DEV[^/]+)/.+", path.upper())
        if m is None:
            raise LabOneQControllerException(
                f"Path '{path}' is not referring to any device"
            )
        serial = m.group(1).lower()
        for dev in self._devices.values():
            if dev.serial == serial:
                return dev
        raise LabOneQControllerException(f"Could not find device for the path '{path}'")

    async def connect(self):
        await self._prepare_daqs()
        self._prepare_devices()
        for device in self._devices.values():
            await device.connect()
        if self._reset_devices:
            await self.load_factory_preset()
        self.start_monitor()
        await self.configure_device_setup()

    async def _configure_parallel(
        self,
        devices: list[DeviceZI],
        control_nodes_getter: Callable[[DeviceZI], list[NodeControlBase]],
        config_name: str,
    ):
        response_waiter = ResponseWaiter()
        set_nodes: list[DaqNodeSetAction] | None = None
        for device in devices:
            dev_nodes = control_nodes_getter(device)

            conditions_checker = ConditionsChecker()
            conditions_checker.add(
                target=device.daq.node_monitor,
                conditions={n.path: n.value for n in filter_conditions(dev_nodes)},
            )
            failed = conditions_checker.check_all()
            if not failed:
                continue

            failed_paths = [path for path, _ in failed]
            wait_conditions = {
                node.path: node.value
                for node in dev_nodes
                if node.kind == NodeControlKind.WaitCondition
                and node.path in failed_paths
            }
            if wait_conditions:
                response_waiter.add(
                    target=device.daq.node_monitor,
                    conditions=wait_conditions,
                )

            if set_nodes is None:
                set_nodes = []
            set_nodes.extend(
                [
                    DaqNodeSetAction(
                        daq=device.daq,
                        path=node.path,
                        value=node.raw_value,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    )
                    for node in filter_commands(dev_nodes)
                ]
            )
            response_waiter.add(
                target=device.daq.node_monitor,
                conditions={n.path: n.value for n in filter_responses(dev_nodes)},
            )

        if set_nodes is None:
            return

        if set_nodes:
            await batch_set(set_nodes)

        timeout = 10
        if not response_waiter.wait_all(timeout=timeout):
            raise LabOneQControllerException(
                f"Internal error: {config_name} for devices "
                f"{[d.dev_repr for d in devices]} is not complete within {timeout}s. "
                f"Not fulfilled:\n{response_waiter.remaining_str()}"
            )

        failed = conditions_checker.check_all()
        if failed:
            raise LabOneQControllerException(
                f"Internal error: {config_name} for devices "
                f"{[d.dev_repr for d in devices]} failed. "
                f"Errors:\n{conditions_checker.failed_str(failed)}"
            )

    async def configure_device_setup(self):
        _logger.info("Configuring the device setup")
        configs = {
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

        self.flush_monitor()  # Ensure status is up-to-date

        leaders = [dev for _, dev in self.leaders]
        if len(leaders) == 0:  # Happens for standalone devices
            leaders = [dev for _, dev in self.followers]
        for config_name, control_nodes_getter in configs.items():
            # Begin by applying a config step from the leader(s), and then proceed with
            # the downstream devices. This is because downstream devices may rely on the
            # settings of upstream devices, such as an external reference supply. If there
            # are any dependencies in the reverse direction, such as the status of a ZSync
            # link, they should be resolved by moving the dependent settings to a later
            # configuration step.
            targets = leaders
            while len(targets) > 0:
                await self._configure_parallel(
                    targets, control_nodes_getter, config_name
                )
                children = []
                for parent_dev in targets:
                    for down_stream_devices in parent_dev._downlinks.values():
                        for _, dev_ref in down_stream_devices:
                            if (dev := dev_ref()) is not None:
                                children.append(dev)
                targets = children
        _logger.info("The device setup is configured")

    def disconnect(self):
        self.reset_monitor()
        for device in self._devices.values():
            device.disconnect()
        self._devices = {}
        self._daqs = {}

    async def disable_outputs(
        self,
        device_uids: list[str] | None = None,
        logical_signals: list[str] | None = None,
        unused_only: bool = False,
    ):
        # Set of outputs to disable or skip (depending on the 'invert' param) per device.
        # Rationale for the logic: the actual number of outputs is only known by the connected
        # device object, here we can only determine the outputs mapped in the device setup.
        outputs_per_device: dict[str, set[int] | None] = {}

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
                device_uid, outputs = self._ds.resolve_ls_path_outputs(ls_path)
                if device_uid is not None:
                    outputs_per_device.setdefault(device_uid, set()).update(outputs)

        all_actions: list[DaqNodeSetAction] = []
        for device_uid, outputs in outputs_per_device.items():
            device = self.find_by_uid(device_uid)
            all_actions.extend(device.disable_outputs(outputs, invert))
        await batch_set(all_actions)

    def shut_down(self):
        for device in self._devices.values():
            device.shut_down()

    def free_allocations(self):
        for device in self._devices.values():
            device.free_allocations()

    async def on_experiment_end(self):
        all_actions: list[DaqNodeSetAction] = []
        for device in self._devices.values():
            all_actions.extend(device.on_experiment_end())
        await batch_set(all_actions)

    def start_monitor(self):
        if self._monitor_started:
            return

        response_waiter = ResponseWaiter()
        for daq in self._daqs.values():
            daq.node_monitor.start()
            response_waiter.add(
                target=daq.node_monitor,
                conditions={path: None for path in daq.node_monitor._nodes},
            )

        if not response_waiter.wait_all(timeout=2):
            raise LabOneQControllerException(
                f"Internal error: Didn't get all the status node values within 2s. "
                f"Missing:\n{response_waiter.remaining_str()}"
            )

        self._monitor_started = True

    def flush_monitor(self):
        for daq in self._daqs.values():
            daq.node_monitor.flush()

    def reset_monitor(self):
        for daq in self._daqs.values():
            daq.node_monitor.reset()
        self._monitor_started = False

    async def load_factory_preset(self):
        reset_nodes = []
        for device in self._devices.values():
            reset_nodes += device.collect_load_factory_preset_nodes()
        await batch_set(reset_nodes)
        for daq in self._daqs.values():
            daq.clear_cache()

    def _validate_dataserver_device_fw_compatibility(self):
        """Validate dataserver and device firmware compatibility."""
        if not self._dry_run and not self._ignore_version_mismatch:
            daq_dev_serials: dict[str, list[str]] = defaultdict(list)
            for device_qualifier in self._ds.instruments:
                daq_dev_serials[device_qualifier.server_uid].append(
                    device_qualifier.options.serial
                )
            for server_uid, dev_serials in daq_dev_serials.items():
                try:
                    check_dataserver_device_compatibility(
                        self._daqs.get(server_uid)._zi_api_object, dev_serials
                    )
                except Exception as error:  # noqa: PERF203
                    raise LabOneQControllerException(str(error)) from error

    def _prepare_devices(self):
        self._validate_dataserver_device_fw_compatibility()

        updated_devices: dict[str, DeviceZI] = {}
        for device_qualifier in self._ds.instruments:
            daq = self._daqs.get(device_qualifier.server_uid)

            if device_qualifier.dry_run:
                cast(DaqWrapperDryRun, daq).map_device_type(device_qualifier)
            device = self._devices.get(device_qualifier.uid)
            if device is None or device.device_qualifier != device_qualifier:
                device = DeviceFactory.create(device_qualifier, daq)
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

        # Make emulated PQSC aware of the down-stream devices
        for device_qualifier in self._ds.instruments:
            if device_qualifier.dry_run and device_qualifier.driver == "PQSC":
                enabled_zsyncs = {}
                from_dev = self._devices[device_qualifier.uid]
                for port, _, to_dev in from_dev.downlinks():
                    if enabled_zsyncs.get(port.lower()) == to_dev.serial:
                        continue
                    enabled_zsyncs[port.lower()] = to_dev.serial
                    cast(DaqWrapperDryRun, daq).set_emulation_option(
                        serial=device_qualifier.options.serial,
                        option=f"{port.lower()}/connection/serial",
                        value=to_dev.serial[3:],
                    )

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

            # Set RF channel offsets
            dev.update_rf_offsets(
                self._ds.get_device_rf_voltage_offsets(device_qualifier.uid)
            )

    async def _prepare_daqs(self):
        updated_daqs: dict[str, DaqWrapper] = {}
        for server_uid, server_qualifier in self._ds.servers:
            existing = self._daqs.get(server_uid)
            if existing is not None and existing.server_qualifier == server_qualifier:
                updated_daqs[server_uid] = existing
                continue

            _logger.info(
                "Connecting to data server at %s:%s",
                server_qualifier.host,
                server_qualifier.port,
            )
            if server_qualifier.dry_run:
                daq = DaqWrapperDryRun(server_uid, server_qualifier)
            else:
                daq = DaqWrapper(server_uid, server_qualifier)
            await daq.validate_connection()
            updated_daqs[server_uid] = daq
        self._daqs = updated_daqs

    def check_errors(self, raise_on_error: bool = True) -> str | None:
        all_errors: list[str] = []
        for _, device in self.all:
            all_errors.extend(decode_errors(device.fetch_errors(), device.dev_repr))
        if len(all_errors) == 0:
            return None
        all_messages = "\n".join(all_errors)
        msg = f"Error(s) happened on device(s) during the execution of the experiment. Error messages:\n{all_messages}"
        if raise_on_error:
            raise LabOneQControllerException(msg)
        return msg


class DeviceErrorSeverity(Enum):
    info = 0
    warning = 1
    error = 2


def decode_errors(errors: list[str], dev_repr: str) -> list[str]:
    collected_messages: list[str] = []
    for error in errors:
        # the key in error["vector"] looks like a dict, but it's a string. so we have to use
        # json.loads to convert it into a dict.
        error_vector = json.loads(error["vector"])
        for message in error_vector["messages"]:
            if message["code"] == "AWGRUNTIMEERROR" and message["params"][0] == 1:
                awg_core = int(message["attribs"][0])
                program_counter = int(message["params"][1])
                collected_messages.append(
                    f"{dev_repr}: Gap detected on AWG core {awg_core}, program counter {program_counter}"
                )
            if message["severity"] >= DeviceErrorSeverity.error.value:
                collected_messages.append(f'{dev_repr}: {message["message"]}')
    return collected_messages
