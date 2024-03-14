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
    DaqWrapper,
    DaqWrapperDryRun,
    batch_set_multiple,
)
from laboneq.controller.devices.async_support import gather_and_apply
from laboneq.controller.devices.device_factory import DeviceFactory
from laboneq.controller.devices.device_setup_dao import DeviceSetupDAO
from laboneq.controller.devices.device_utils import (
    NodeCollector,
    prepare_emulator_state,
)
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.devices.zi_emulator import EmulatorState
from laboneq.controller.devices.zi_node_monitor import (
    ConditionsChecker,
    NodeControlBase,
    ResponseWaiter,
    filter_commands,
    filter_settings,
    filter_conditions,
    filter_responses,
    filter_wait_conditions,
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
        ignore_version_mismatch: bool = False,
    ):
        self._ds = DeviceSetupDAO(
            target_setup=target_setup,
            ignore_version_mismatch=ignore_version_mismatch,
        )
        self._emulator_state: EmulatorState | None = None
        self._ignore_version_mismatch = ignore_version_mismatch
        self._daqs: dict[str, DaqWrapper] = {}
        self._devices: dict[str, DeviceZI] = {}
        self._monitor_started = False

    @property
    def emulator_state(self) -> EmulatorState:
        if self._emulator_state is None:
            self._emulator_state = prepare_emulator_state(self._ds)
        return self._emulator_state

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

    @property
    def with_node_monitor(self) -> Iterator[DeviceZI]:
        for device in self._devices.values():
            if device._node_monitor is not None:
                yield device

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
        for _, dev in self.all:
            if dev.serial == serial:
                return dev
        raise LabOneQControllerException(f"Could not find device for the path '{path}'")

    async def connect(self, do_emulation: bool, reset_devices: bool = False):
        await self._prepare_daqs(do_emulation=do_emulation)
        if not do_emulation:
            self._validate_dataserver_device_fw_compatibility()  # TODO(2K): Uses zhinst utils -> async api version?
        self._prepare_devices()
        for _, device in self.all:
            await device.connect(self.emulator_state if do_emulation else None)
        await self.start_monitor()
        await self.configure_device_setup(reset_devices)

    async def _configure_parallel(
        self,
        devices: list[DeviceZI],
        control_nodes_getter: Callable[[DeviceZI], list[NodeControlBase]],
        config_name: str,
    ):
        conditions_checker = ConditionsChecker()
        response_waiter = ResponseWaiter()
        set_nodes: dict[DeviceZI, NodeCollector] = defaultdict(NodeCollector)

        def _add_set_nodes(device: DeviceZI, nodes: list[NodeControlBase]):
            for node in nodes:
                set_nodes[device].add(
                    path=node.path,
                    value=node.raw_value,
                    cache=False,
                )

        for device in devices:
            dev_nodes = control_nodes_getter(device)

            commands = filter_commands(dev_nodes)
            if len(commands) > 0:
                # 1a. Unconditional command
                _add_set_nodes(device, commands)
                response_waiter.add(
                    target=device,
                    conditions={n.path: n.value for n in filter_responses(dev_nodes)},
                )
            else:
                # 1b. Verify if device is already configured as desired
                dev_conditions_checker = ConditionsChecker()
                dev_conditions_checker.add(
                    target=device,
                    conditions={n.path: n.value for n in filter_conditions(dev_nodes)},
                )
                conditions_checker.add_from(dev_conditions_checker)
                failed = dev_conditions_checker.check_all()
                if not failed:
                    continue

                failed_paths = [path for path, _ in failed]
                failed_nodes = [n for n in dev_nodes if n.path in failed_paths]
                response_waiter.add(
                    target=device,
                    conditions={
                        n.path: n.value for n in filter_wait_conditions(failed_nodes)
                    },
                )

                _add_set_nodes(device, filter_settings(dev_nodes))
                response_waiter.add(
                    target=device,
                    conditions={n.path: n.value for n in filter_responses(dev_nodes)},
                )

        # 2. Apply any necessary node changes (which may be empty)
        if len(set_nodes) > 0:
            async with gather_and_apply(batch_set_multiple) as awaitables:
                for device, nc in set_nodes.items():
                    awaitables.append(device.maybe_async(nc))

        # 3. Wait for responses to the changes in step 2 and settling of conditions resulting from earlier config stages
        if len(response_waiter.remaining()) == 0:
            # Nothing to wait for, e.g. all devices were already configured as desired?
            return

        timeout = 10
        if not await response_waiter.wait_all(timeout=timeout):
            raise LabOneQControllerException(
                f"Internal error: {config_name} for devices "
                f"{[d.dev_repr for d in devices]} is not complete within {timeout}s. "
                f"Not fulfilled:\n{response_waiter.remaining_str()}"
            )

        # 4. Recheck all the conditions, as some may have potentially changed as a result of step 2
        failed = conditions_checker.check_all()
        if failed:
            raise LabOneQControllerException(
                f"Internal error: {config_name} for devices "
                f"{[d.dev_repr for d in devices]} failed. "
                f"Errors:\n{conditions_checker.failed_str(failed)}"
            )

    async def configure_device_setup(self, reset_devices: bool):
        _logger.info("Configuring the device setup")

        await self.flush_monitor()  # Ensure status is up-to-date

        if reset_devices:
            await self._configure_parallel(
                [d for d in self._devices.values() if not d.is_secondary],
                lambda d: cast(DeviceZI, d).load_factory_preset_control_nodes(),
                "Reset to factory defaults",
            )
            await (
                self.flush_monitor()
            )  # Consume any updates resulted from the above reset
            # TODO(2K): Error check
            for _, device in self.all:
                device.clear_cache()

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

    async def disconnect(self):
        await self.reset_monitor()
        for device in self._devices.values():
            device.disconnect()
        self._devices = {}
        self._daqs = {}
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
                device_uid, outputs = self._ds.resolve_ls_path_outputs(ls_path)
                if device_uid is not None:
                    outputs_per_device[device_uid].update(outputs)

        async with gather_and_apply(batch_set_multiple) as awaitables:
            for device_uid, outputs in outputs_per_device.items():
                device = self.find_by_uid(device_uid)
                awaitables.append(device.disable_outputs(outputs, invert))

    def free_allocations(self):
        for device in self._devices.values():
            device.free_allocations()

    async def on_experiment_end(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for device in self._devices.values():
                awaitables.append(device.maybe_async(device.on_experiment_end()))

    async def start_monitor(self):
        if self._monitor_started:
            return

        response_waiter = ResponseWaiter()
        for device in self.with_node_monitor:
            await device.node_monitor.start()
            response_waiter.add(
                target=device,
                conditions={path: None for path in device.nodes_to_monitor()},
            )

        if not await response_waiter.wait_all(timeout=2):
            raise LabOneQControllerException(
                f"Internal error: Didn't get all the status node values within 2s. "
                f"Missing:\n{response_waiter.remaining_str()}"
            )

        self._monitor_started = True

    async def flush_monitor(self):
        for node_monitor in {device.node_monitor for device in self.with_node_monitor}:
            await node_monitor.flush()

    async def reset_monitor(self):
        for node_monitor in {device.node_monitor for device in self.with_node_monitor}:
            await node_monitor.reset()
        self._monitor_started = False

    def _validate_dataserver_device_fw_compatibility(self):
        """Validate dataserver and device firmware compatibility."""
        if not self._ignore_version_mismatch:
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
        updated_devices: dict[str, DeviceZI] = {}
        for device_qualifier in self._ds.instruments:
            device = self._devices.get(device_qualifier.uid)
            if device is None or device.device_qualifier != device_qualifier:
                daq = self._daqs[device_qualifier.server_uid]
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

    async def _prepare_daqs(self, do_emulation: bool):
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
            daq: DaqWrapper
            if do_emulation:
                daq = DaqWrapperDryRun(
                    server_uid, server_qualifier, self.emulator_state
                )
            else:
                daq = DaqWrapper(server_uid, server_qualifier)
            await daq.validate_connection()
            updated_daqs[server_uid] = daq
        self._daqs = updated_daqs

    async def check_errors(self, raise_on_error: bool = True) -> str | None:
        all_errors: list[str] = []
        for _, device in self.all:
            dev_errors = await device.fetch_errors()
            all_errors.extend(decode_errors(dev_errors, device.dev_repr))
        if len(all_errors) == 0:
            return None
        all_messages = "\n".join(all_errors)
        msg = f"Error(s) happened on device(s) during the execution of the experiment. Error messages:\n{all_messages}"
        if raise_on_error:
            raise LabOneQControllerException(msg)
        return msg

    async def update_warning_nodes(self):
        for node_monitor in {device.node_monitor for device in self.with_node_monitor}:
            await node_monitor.poll()

        for device in self.with_node_monitor:
            device.update_warning_nodes(
                {
                    node: device.node_monitor.get_last(node)
                    for node in device.collect_warning_nodes()
                }
            )


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
