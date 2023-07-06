# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum

from laboneq.controller.communication import CachingStrategy, DaqNodeSetAction
from laboneq.controller.devices.device_zi import DeviceZI
from laboneq.controller.devices.zi_node_monitor import (
    Command,
    Condition,
    NodeControlBase,
    Response,
)


class ReferenceClockSourceSHF(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1
    ZSYNC = 2


class DeviceSHFBase(DeviceZI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reference_clock_source = ReferenceClockSourceSHF.ZSYNC

    def update_clock_source(self, force_internal: bool | None):
        if self.is_standalone() and force_internal is not False:
            # Internal is the default (or explicit) for standalone SHF
            self._reference_clock_source = ReferenceClockSourceSHF.INTERNAL
        elif self.is_standalone() and force_internal is not True:
            # External specified explicitly for standalone SHF
            self._reference_clock_source = ReferenceClockSourceSHF.EXTERNAL
        else:
            # ZSync is the only possible source when device is not standalone
            self._reference_clock_source = ReferenceClockSourceSHF.ZSYNC

    def collect_load_factory_preset_nodes(self):
        return [
            DaqNodeSetAction(
                self._daq,
                f"/{self.serial}/system/preset/load",
                1,
                caching_strategy=CachingStrategy.NO_CACHE,
            )
        ]

    def clock_source_control_nodes(self) -> list[NodeControlBase]:
        expected_freq = {
            ReferenceClockSourceSHF.INTERNAL: None,
            ReferenceClockSourceSHF.EXTERNAL: 10e6,
            ReferenceClockSourceSHF.ZSYNC: 100e6,
        }[self._reference_clock_source]
        source = self._reference_clock_source.value

        return [
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/freq", expected_freq
            ),
            Command(f"/{self.serial}/system/clocks/referenceclock/in/source", source),
            Response(
                f"/{self.serial}/system/clocks/referenceclock/in/sourceactual", source
            ),
            Response(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
        ]
