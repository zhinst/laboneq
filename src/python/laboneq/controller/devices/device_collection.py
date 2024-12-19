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
    DaqWrapperDummy,
    batch_set,
    batch_set_multiple,
)
from laboneq.controller.devices.async_support import (
    ConditionsCheckerAsync,
    _gather,
    _gather_with_timeout,
    async_check_dataserver_device_compatibility,
    gather_and_apply,
    wait_for_state_change,
)
from laboneq.controller.devices.device_factory import DeviceFactory
from laboneq.controller.devices.device_qhub import DeviceQHUB
from laboneq.controller.devices.device_setup_dao import DeviceSetupDAO
from laboneq.controller.devices.device_utils import (
    NodeCollector,
    prepare_emulator_state,
)
from laboneq.controller.devices.device_zi import DeviceBase, DeviceZI
from laboneq.controller.devices.zi_emulator import EmulatorState
from laboneq.controller.devices.zi_node_monitor import (
    ConditionsChecker,
    NodeControlBase,
    NodeControlKind,
    NodeMonitorBase,
    ResponseWaiter,
    filter_commands,
    filter_settings,
    filter_states,
    filter_responses,
    filter_wait_conditions,
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
        self._daqs: dict[str, DaqWrapper] = {}
        self._devices: dict[str, DeviceZI] = {}
        self._monitor_started = False

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
    def node_monitors(self) -> Iterator[NodeMonitorBase]:
        already_considered = set()
        for _, device in self.all:
            node_monitor = device.node_monitor
            if node_monitor not in already_considered:
                already_considered.add(node_monitor)
                yield node_monitor

    @property
    def has_qhub(self) -> bool:
        return self._ds.has_qhub

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
        use_async_api: bool = False,
        disable_runtime_checks: bool = False,
    ):
        await self._prepare_daqs(do_emulation=do_emulation, use_async_api=use_async_api)
        await self._validate_dataserver_device_fw_compatibility(
            emulator_state=self.emulator_state if do_emulation else None,
            use_async_api=use_async_api,
            timeout_s=self._timeout_s,
        )
        self._prepare_devices()
        for _, device in self.all:
            await device.connect(
                self.emulator_state if do_emulation else None,
                use_async_api=use_async_api,
                disable_runtime_checks=disable_runtime_checks,
                timeout_s=self._timeout_s,
            )
        await self.start_monitor()
        await self.configure_device_setup(reset_devices, use_async_api=use_async_api)

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

            # 1. Collect all nodes that are not in the desired state for the device
            dev_conditions_checker = ConditionsChecker()
            dev_conditions_checker.add(
                target=device,
                conditions={n.path: n.value for n in filter_states(dev_nodes)},
            )
            conditions_checker.add_from(dev_conditions_checker)
            failed = dev_conditions_checker.check_all()

            commands = filter_commands(dev_nodes)
            if len(commands) > 0:
                # 1a. Has unconditional commands? Use simplified flow.
                _add_set_nodes(device, commands)
                response_waiter.add(
                    target=device,
                    conditions={n.path: n.value for n in filter_responses(dev_nodes)},
                )
                if failed:
                    response_waiter.add(
                        target=device,
                        conditions={
                            n.path: n.value for n in filter_wait_conditions(dev_nodes)
                        },
                    )
            else:
                # 1b. Is device already in the desired state?
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

        # 3. Wait for responses to the changes in step 2 and settling of dependent states
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

    async def configure_device_setup(self, reset_devices: bool, use_async_api: bool):
        _logger.info("Configuring the device setup")

        configure = self._configure_async if use_async_api else self._configure_parallel

        await self.flush_monitor()  # Ensure status is up-to-date

        def _followers_of(parents: list[DeviceZI]) -> list[DeviceZI]:
            children = []
            for parent_dev in parents:
                for down_stream_devices in parent_dev._downlinks.values():
                    for _, dev_ref in down_stream_devices:
                        if (dev := dev_ref()) is not None:
                            children.append(dev)
            return children

        if reset_devices:
            await configure(
                [device for _, device in self.all if not device.is_secondary],
                lambda d: cast(DeviceZI, d).load_factory_preset_control_nodes(),
                "Reset to factory defaults",
            )
            # Consume any updates resulted from the above reset
            await self.flush_monitor()
            # TODO(2K): Error check
            for _, device in self.all:
                device.clear_cache()

        if self.has_qhub:
            leaders = [dev for _, dev in self.leaders]
            followers = _followers_of(leaders)
            # Switch QHUB to the external clock as usual
            await configure(
                leaders,
                lambda d: cast(DeviceZI, d).clock_source_control_nodes(),
                "QHUB: Reference clock switching",
            )
            # Check if zsync link is already established
            if use_async_api:
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
            else:
                check_zsync_link = ConditionsChecker()
                for dev in followers:
                    check_zsync_link.add(
                        target=dev,
                        conditions={
                            n.path: n.value
                            for n in filter_states(dev.clock_source_control_nodes())
                        },
                    )
                for dev in leaders:
                    check_zsync_link.add(
                        target=dev,
                        conditions={
                            n.path: n.value
                            for n in filter_states(dev.zsync_link_control_nodes())
                        },
                    )
                failed = check_zsync_link.check_all()
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
                        await batch_set(await dev.maybe_async(nc))
                # Reset the QHub phy. TODO(2K): This is the actual workaround, remove this whole block once fixed.
                for dev in leaders:
                    if isinstance(dev, DeviceQHUB):
                        await dev.qhub_reset_zsync_phy()
                # Wait for completion using the dumb get method, as the streamed state sequence may be unreliable
                # with this workaround due to unexpected state bouncing.
                if use_async_api:
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
                else:
                    for (
                        node_monitor,
                        daq_conditions,
                    ) in check_zsync_link._conditions.items():
                        for path, expected in daq_conditions.items():
                            await node_monitor.wait_for_state_by_get(path, expected)
                    # Consume any updates resulted from the above manipulations
                    await self.flush_monitor()
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
                await configure(targets, control_nodes_getter, config_name)
                targets = _followers_of(targets)
        _logger.info("The device setup is configured")

    async def disconnect(self):
        await self.reset_monitor()
        for _, device in self.all:
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
                maybe_device_uid, outputs = self._ds.resolve_ls_path_outputs(ls_path)
                if maybe_device_uid is not None:
                    outputs_per_device[maybe_device_uid].update(outputs)

        async with gather_and_apply(batch_set_multiple) as awaitables:
            for device_uid, outputs in outputs_per_device.items():
                device = self.find_by_uid(device_uid)
                awaitables.append(device.disable_outputs(outputs, invert))

    def free_allocations(self):
        for _, device in self.all:
            device.free_allocations()

    async def on_experiment_begin(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self.all:
                awaitables.append(device.on_experiment_begin())
        await self._update_warning_nodes(init=True)

    async def on_after_nt_step(self):
        await self._update_warning_nodes()
        device_errors = await self.fetch_device_errors()
        if device_errors is not None:
            raise LabOneQControllerException(device_errors)

    async def on_experiment_end(self):
        async with gather_and_apply(batch_set_multiple) as awaitables:
            for _, device in self.all:
                awaitables.append(device.on_experiment_end())

    async def start_monitor(self):
        if self._monitor_started:
            return
        await _gather(*(node_monitor.start() for node_monitor in self.node_monitors))

        response_waiter = ResponseWaiter()
        for _, device in self.all:
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
        await _gather(*(node_monitor.flush() for node_monitor in self.node_monitors))

    async def poll_monitor(self):
        await _gather(*(node_monitor.poll() for node_monitor in self.node_monitors))

    async def reset_monitor(self):
        await _gather(*(node_monitor.reset() for node_monitor in self.node_monitors))
        self._monitor_started = False

    async def _validate_dataserver_device_fw_compatibility(
        self,
        emulator_state: EmulatorState | None,
        use_async_api: bool,
        timeout_s: float,
    ):
        """Validate dataserver and device firmware compatibility."""

        daq_dev_serials: dict[str, list[str]] = defaultdict(list)
        for device_qualifier in self._ds.instruments:
            daq_dev_serials[device_qualifier.server_uid].append(
                device_qualifier.options.serial
            )

        for server_uid, dev_serials in daq_dev_serials.items():
            if use_async_api:
                server_qualifier = next(
                    s[1] for s in self._ds.servers if s[0] == server_uid
                )
                await async_check_dataserver_device_compatibility(
                    server_qualifier.host,
                    server_qualifier.port,
                    dev_serials,
                    emulator_state,
                    self._ignore_version_mismatch,
                    timeout_s,
                )
            elif emulator_state is None and not self._ignore_version_mismatch:
                try:
                    check_dataserver_device_compatibility(
                        self._daqs[server_uid]._zi_api_object, dev_serials
                    )
                except Exception as error:  # noqa: PERF203
                    raise LabOneQControllerException(str(error)) from error

    def _prepare_devices(self):
        updated_devices: dict[str, DeviceZI] = {}
        for device_qualifier in self._ds.instruments:
            device = self._devices.get(device_qualifier.uid)
            if device is None or device.device_qualifier != device_qualifier:
                daq = self._daqs[device_qualifier.server_uid]
                device = DeviceFactory.create(
                    device_qualifier, daq, self._ds.setup_caps
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

    async def _prepare_daqs(self, do_emulation: bool, use_async_api: bool):
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
            if use_async_api:
                daq = DaqWrapperDummy(server_qualifier)
            else:
                _logger.warning(
                    "\n\n"
                    "====================================================================================\n"
                    "The synchronous API you enabled is deprecated for LabOne Q and will soon be removed.\n"
                    "Please contact Zurich Instruments about the use case for the synchronous API, unless\n"
                    "it's a known limitation regarding the missing node log, which is set to be fixed.\n"
                    "====================================================================================\n"
                )
                if do_emulation:
                    daq = DaqWrapperDryRun(
                        server_uid, server_qualifier, self.emulator_state
                    )
                else:
                    daq = DaqWrapper(server_uid, server_qualifier)
                await daq.validate_connection()
            updated_daqs[server_uid] = daq
        self._daqs = updated_daqs

    async def _update_warning_nodes(self, init: bool = False):
        await self.poll_monitor()

        for _, device in self.all:
            await device.update_warning_nodes(init)

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
                    f'{dev_repr}: ({message["code"]}) {message["message"]}'
                )
    return collected_messages
