# Copyright 2019 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from enum import IntEnum
import time

from laboneq.controller.devices.async_support import ResponseWaiterAsync, _sleep
from laboneq.controller.devices.device_utils import NodeCollector
from laboneq.controller.devices.device_zi import DeviceBase
from laboneq.controller.devices.node_control import (
    Setting,
    Condition,
    NodeControlBase,
    Response,
)
from laboneq.controller.recipe_processor import RecipeData
from laboneq.controller.utilities.exception import LabOneQControllerException
from laboneq.data.recipe import NtStepKey

_logger = logging.getLogger(__name__)


class ReferenceClockSourceLeader(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1


class DeviceLeaderBase(DeviceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_internal_clock = False
        self._zsyncs = 0
        self._downstream_serial_to_zsync: dict[str, int] = {}
        self._downstream_uid_to_serial: dict[str, str] = {}

    @property
    def zsyncs(self) -> int:
        return self._zsyncs

    def update_clock_source(self, force_internal: bool | None):
        self._use_internal_clock = force_internal is True

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        source = (
            ReferenceClockSourceLeader.INTERNAL
            if self._use_internal_clock
            else ReferenceClockSourceLeader.EXTERNAL
        )
        expected_freq = None if self._use_internal_clock else 10e6
        return [
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/freq", expected_freq
            ),
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/sourceactual", source
            ),
            Setting(f"/{self.serial}/system/clocks/referenceclock/in/source", source),
            Response(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
        ]

    async def wait_for_zsync_link(self, timeout_s: float):
        self._downstream_serial_to_zsync = {}
        self._downstream_uid_to_serial = {}
        expected_serials: set[str] = set()
        unknown_serials: set[str] = set()
        for dev_ref in self._downlinks:
            dev = dev_ref()
            if dev is None:
                continue
            expected_serials.add(dev.serial)
            self._downstream_uid_to_serial[dev.device_qualifier.uid] = dev.serial
        nodes = [
            *[
                f"/{self.serial}/zsyncs/{zsync}/connection/status"
                for zsync in range(self.zsyncs)
            ],
            *[
                f"/{self.serial}/zsyncs/{zsync}/connection/serial"
                for zsync in range(self.zsyncs)
            ],
        ]
        start_time: float | None = None
        while True:
            results = await self._api.get_raw(nodes)
            for zsync in range(self.zsyncs):
                status = results[f"/{self.serial}/zsyncs/{zsync}/connection/status"]
                if status != 2:  # not connected?
                    continue
                serial_no = results[f"/{self.serial}/zsyncs/{zsync}/connection/serial"]
                if len(serial_no) == 0:
                    # TODO(2K): Emulator returns connected status for all ports
                    continue
                serial = f"dev{serial_no}"
                if serial in expected_serials:
                    self._downstream_serial_to_zsync[serial] = zsync
                else:
                    unknown_serials.add(serial)
            if set(self._downstream_serial_to_zsync.keys()) == expected_serials:
                break
            if start_time is None:
                start_time = time.monotonic()
            elif time.monotonic() - start_time > timeout_s:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Timeout waiting for ZSync link to be established. "
                    f"Connected devices: {set(self._downstream_serial_to_zsync.keys())}. "
                    f"Not connected devices: {expected_serials - set(self._downstream_serial_to_zsync.keys())}. "
                    f"Unknown devices: {unknown_serials}"
                )
            await _sleep(0.1)
        if len(unknown_serials) > 0:
            _logger.warning(
                f"{self.dev_repr}: Unknown devices connected to ZSync: {unknown_serials}"
            )

    async def configure_feedback(self, recipe_data: RecipeData):
        nc = NodeCollector(base=f"/{self.serial}/")
        min_wait_time = recipe_data.recipe.max_step_execution_time
        # This is required because PQSC/QHUB is only receiving the feedback events
        # during the holdoff time, even for a single trigger.
        nc.add("execution/holdoff", min_wait_time)
        enabled_zsyncs = set()
        for awg_key, awg_config in recipe_data.awg_configs.items():
            if awg_config.source_feedback_register in (None, "local"):
                continue  # Only consider devices receiving feedback from PQSC/QHUB
            downstream_serial = self._downstream_uid_to_serial.get(awg_key.device_uid)
            if (
                downstream_serial is None
                or downstream_serial not in self._downstream_serial_to_zsync
            ):
                continue  # Unknown device or not connected to ZSync
            p_addr = self._downstream_serial_to_zsync[downstream_serial]
            zsync_output = f"zsyncs/{p_addr}/output"
            zsync_base = f"{zsync_output}/registerbank"
            if p_addr not in enabled_zsyncs:
                nc.add(f"{zsync_output}/enable", 1)
                nc.add(f"{zsync_output}/source", 0)
            enabled_zsyncs.add(p_addr)

            reg_selector_base = f"{zsync_base}/sources/{awg_config.fb_reg_target_index}"
            nc.add(f"{reg_selector_base}/enable", 1)
            nc.add(
                f"{reg_selector_base}/register",
                awg_config.source_feedback_register,
            )
            nc.add(f"{reg_selector_base}/index", awg_config.fb_reg_source_index)
        await self.set_async(nc)

    async def start_execution(self, with_pipeliner: bool):
        _logger.debug("Starting execution...")
        nc = NodeCollector(base=f"/{self.serial}/")

        nc.add("triggers/out/0/enable", 1, cache=False)

        # Select the first ZSYNC port from downlinks.

        p_addr = next(
            (port for port in self._downstream_serial_to_zsync.values()),
            None,
        )
        # Do not touch any setting if there is no valid ZSYNC connection
        if p_addr is not None:
            # Why do we set this?
            # Trigger output on PQSC/QHUB mirrors the ZSYNC signals sent through
            # `p_addr`. No valid connection -> no ZSYNC signal -> no trigger output
            # -> (probably nothing on the scope because trigger out is typically connected to a scope)
            nc.add("triggers/out/0/port", p_addr, cache=False)

        nc.add("execution/enable", 1, cache=False)

        await self.set_async(nc)

    async def setup_one_step_execution(
        self, recipe_data: RecipeData, nt_step: NtStepKey, with_pipeliner: bool
    ):
        if with_pipeliner:
            # TODO(2K): Use timeout from connect
            rw = ResponseWaiterAsync(
                api=self._api, dev_repr=self.dev_repr, timeout_s=1.0
            )
            rw.add_nodes(
                {
                    f"/{self.serial}/execution/synchronization/enable": 1  # sync enabled
                }
            )
            await rw.prepare()
            nc = NodeCollector(base=f"/{self.serial}/")
            nc.add("execution/synchronization/enable", 1)
            await self.set_async(nc)
            if len(await rw.wait()) > 0:
                raise LabOneQControllerException(
                    f"{self.dev_repr}: Internal error: Failed to enable synchronization"
                )

    async def teardown_one_step_execution(self, with_pipeliner: bool):
        nc = NodeCollector(base=f"/{self.serial}/")
        if with_pipeliner:
            nc.add("execution/synchronization/enable", 0)
        await self.set_async(nc)

    async def configure_trigger(self, recipe_data: RecipeData):
        initialization = recipe_data.get_initialization(self.uid)
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("system/clocks/referenceclock/out/enable", 1)
        nc.add("execution/repetitions", initialization.config.repetitions)
        await self.set_async(nc)

    async def reset_to_idle(self):
        await super().reset_to_idle()
        nc = NodeCollector(base=f"/{self.serial}/")
        nc.add("execution/synchronization/enable", 0, cache=False)
        await self.set_async(nc)
