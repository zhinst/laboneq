# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

from laboneq.controller.communication import (
    CachingStrategy,
    DaqNodeSetAction,
    DaqWrapper,
    DaqWrapperDryRun,
    ServerQualifier,
    batch_set,
)
from laboneq.controller.devices.device_factory import DeviceFactory
from laboneq.controller.devices.device_zi import DeviceQualifier, DeviceZI
from laboneq.controller.devices.zi_node_monitor import (
    ConditionsChecker,
    ResponseWaiter,
    filter_commands,
    filter_conditions,
    filter_responses,
)
from laboneq.controller.util import LabOneQControllerException
from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.servers import DataServer


class DeviceCollection:
    def __init__(
        self,
        device_setup: DeviceSetup,
        dry_run: bool,
        ignore_lab_one_version_error: bool = False,
    ):
        self._device_setup: DeviceSetup = deepcopy(device_setup)
        self._dry_run = dry_run
        self._ignore_lab_one_version_error = ignore_lab_one_version_error
        self._daqs: Dict[str, DaqWrapper] = {}
        self._devices: Dict[str, DeviceZI] = {}
        self._logger = logging.getLogger(__name__)

    @property
    def ds_instruments(self) -> Iterator[ZIStandardInstrument]:
        for instrument in self._device_setup.instruments:
            if isinstance(instrument, ZIStandardInstrument):
                yield instrument

    @property
    def all(self) -> Iterator[Tuple[str, DeviceZI]]:
        for uid, device in self._devices.items():
            yield uid, device

    @property
    def leaders(self) -> Iterator[Tuple[str, DeviceZI]]:
        for uid, device in self._devices.items():
            if device.is_leader():
                yield uid, device

    @property
    def followers(self) -> Iterator[Tuple[str, DeviceZI]]:
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

    def find_by_path(self, path: str):
        m = re.match(r"^/?(DEV\d+)/.+", path.upper())
        if m is None:
            raise LabOneQControllerException(
                f"Path '{path}' is not referring to any device"
            )
        serial = m.group(1).lower()
        for dev in self._devices.values():
            if dev.serial == serial:
                return dev
        raise LabOneQControllerException(f"Could not find device for the path '{path}'")

    def connect(self, is_using_standalone_compiler: bool = True):
        self._prepare_daqs()
        self._prepare_devices(is_using_standalone_compiler)
        for device in self._devices.values():
            device.connect()
        self.start_monitor()
        self.init_clocks()
        self.stop_monitor()

    def _configure_clocks_parallel(self, devices: List[DeviceZI]):
        conditions_checker = ConditionsChecker()
        response_waiter = ResponseWaiter()
        set_clock_nodes: List[DaqNodeSetAction] = []
        for device in devices:
            dev_nodes = device.clock_source_control_nodes()
            set_clock_nodes.extend(
                [
                    DaqNodeSetAction(
                        daq=device.daq,
                        path=path,
                        value=val,
                        caching_strategy=CachingStrategy.NO_CACHE,
                    )
                    for path, val in filter_commands(dev_nodes).items()
                ]
            )
            conditions_checker.add(
                target=device.daq.node_monitor,
                conditions=filter_conditions(dev_nodes),
            )
            response_waiter.add(
                target=device.daq.node_monitor,
                conditions=filter_responses(dev_nodes),
            )

        failed_path, _ = conditions_checker.check_all()
        if failed_path is None:
            return

        batch_set(set_clock_nodes)
        timeout = 5
        if not response_waiter.wait_all(timeout=timeout):
            raise RuntimeError(
                f"Internal error: Reference clock switching for devices "
                f"{[d.dev_repr for d in devices]} is not complete within {timeout}s. "
                f"Not fulfilled:\n{response_waiter.remaining_str()}"
            )

        failed_path, expected = conditions_checker.check_all()
        if failed_path is not None:
            raise RuntimeError(
                f"Internal error: Reference clock switching for devices "
                f"{[d.dev_repr for d in devices]} failed at {failed_path} != {expected}."
            )

    def init_clocks(self):
        self._logger.info("Configuring clock sources")
        # Wait until clock status is available for all devices
        response_waiter = ResponseWaiter()
        for device in self._devices.values():
            target_node_monitor = device.daq.node_monitor
            clock_source_control_nodes = [
                node.path for node in device.clock_source_control_nodes()
            ]
            target_node_monitor.fetch(clock_source_control_nodes)
            response_waiter.add(
                target=target_node_monitor,
                conditions={path: None for path in clock_source_control_nodes},
            )

        if not response_waiter.wait_all(timeout=2):
            raise LabOneQControllerException(
                f"Internal error: Didn't get all the clock status node values within 2s. "
                f"Missing:\n{response_waiter.remaining_str()}"
            )

        # Begin switching extrefs from the leaders, and then by downstream links,
        # as downstream device may get its extref from the upstream one.
        parents = [dev for _, dev in self.leaders]
        if len(parents) == 0:  # Happens for standalone devices
            parents = [dev for _, dev in self.followers]
        while len(parents) > 0:
            self._configure_clocks_parallel(parents)
            children = []
            for parent_dev in parents:
                for _, dev_ref in parent_dev._downlinks.items():
                    dev = dev_ref()
                    if dev is not None:
                        children.append(dev)
            parents = children
        self._logger.info("Clock sources configured")

    def disconnect(self):
        self.stop_monitor()
        for device in self._devices.values():
            device.disconnect()
        self._devices = {}
        self._daqs = {}

    def shut_down(self):
        for device in self._devices.values():
            device.shut_down()

    def free_allocations(self):
        for device in self._devices.values():
            device.free_allocations()

    def start_monitor(self):
        for daq in self._daqs.values():
            daq.node_monitor.stop()
            daq.node_monitor.start()

    def flush_monitor(self):
        for daq in self._daqs.values():
            daq.node_monitor.flush()

    def stop_monitor(self):
        for daq in self._daqs.values():
            daq.node_monitor.stop()

    def _prepare_devices(self, is_using_standalone_compiler):
        def make_device_qualifier(
            instrument: ZIStandardInstrument, daq: DaqWrapper
        ) -> DeviceQualifier:
            driver = instrument.calc_driver()
            options = {
                **instrument.calc_options(),
                "standalone_awg": is_using_standalone_compiler,
            }
            if len(instrument.connections) == 0:
                # Treat devices without connections as non-QC
                if "dev_type" not in options:
                    options["dev_type"] = driver
                if options.get("is_qc", False):
                    options["is_qc"] = False
                driver = "NONQC"

            return DeviceQualifier(
                dry_run=self._dry_run, driver=driver, server=daq, options=options
            )

        updated_devices: Dict[str, DeviceZI] = {}
        for instrument in self.ds_instruments:
            daq = self._daqs.get(instrument.server_uid)
            device_qualifier = make_device_qualifier(instrument, daq)
            device = self._devices.get(instrument.uid)
            if device is None or device.device_qualifier != device_qualifier:
                device = DeviceFactory.create(device_qualifier)
            device.remove_all_links()
            updated_devices[instrument.uid] = device
        self._devices = updated_devices

        # Update device links and leader/follower status
        for instrument in self.ds_instruments:
            from_dev = self._devices[instrument.uid]
            for connection in instrument.connections:
                if connection.signal_type in [IOSignalType.DIO, IOSignalType.ZSYNC]:
                    from_port = connection.local_port
                    to_dev = self._devices.get(connection.remote_path)
                    if to_dev is None:
                        raise LabOneQControllerException(
                            f"Could not find destination device '{connection.remote_path}' for "
                            f"the port '{connection.local_port}' connection of the "
                            f"device '{instrument.uid}'"
                        )
                    to_port = f"{connection.signal_type.name}/{connection.remote_port}"
                    from_dev.add_downlink(from_port, to_dev)
                    to_dev.add_uplink(to_port, from_dev)

        # Set clock source (external by default)
        for instrument in self.ds_instruments:
            dev = self._devices[instrument.uid]
            force_internal: Optional[bool] = None
            if instrument.reference_clock_source is not None:
                force_internal = (
                    instrument.reference_clock_source == ReferenceClockSource.INTERNAL
                )
            dev.update_clock_source(force_internal)

    def _prepare_daqs(self):
        def make_server_qualifier(server: DataServer):
            return ServerQualifier(
                dry_run=self._dry_run,
                host=server.host,
                port=int(server.port),
                api_level=int(server.api_level),
                ignore_lab_one_version_error=self._ignore_lab_one_version_error,
            )

        updated_daqs: Dict[str, DaqWrapper] = {}
        for server_uid, server in self._device_setup.servers.items():
            server_qualifier = make_server_qualifier(server)
            existing = self._daqs.get(server_uid)
            if existing is not None and existing.server_qualifier == server_qualifier:
                existing.node_monitor.reset()
                updated_daqs[server_uid] = existing
                continue

            self._logger.info(
                "Connecting to data server at %s:%s", server.host, server.port
            )
            if server_qualifier.dry_run:
                daq = DaqWrapperDryRun(server_uid, server_qualifier)
                for instr in self.ds_instruments:
                    if instr.server_uid == server_uid:
                        daq.map_device_type(
                            instr.address, instr.calc_driver(), instr.calc_options()
                        )
            else:
                daq = DaqWrapper(server_uid, server_qualifier)
            updated_daqs[server_uid] = daq
        self._daqs = updated_daqs
