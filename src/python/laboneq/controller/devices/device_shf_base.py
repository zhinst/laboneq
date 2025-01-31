# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import IntEnum

from laboneq.controller.devices.device_zi import DeviceBase
from laboneq.controller.devices.node_control import (
    Command,
    Setting,
    Condition,
    NodeControlBase,
    Response,
    WaitCondition,
)

import logging

_logger = logging.getLogger(__name__)

OPT_SHF_PLUS = "PLUS"


def check_synth_frequency(synth_cf: float, device: str, index: int):
    if abs(synth_cf % 200e6) > 1e-6:
        _logger.warning(
            f"Setting center frequency on device {device}, synthesizer {index} to"
            f" {synth_cf / 1e9:.3} GHz.\n"
            "To ensure reproducible phase relationships between different channels of"
            " the same instrument or when switching between RF center frequencies on"
            " the same channel, it is necessary to use an integer multiple of 200 MHz."
            " For this reason, it is strongly recommended to use multiples of 200 MHz"
            " by default for the center frequencies of all experiments."
        )


class ReferenceClockSourceSHF(IntEnum):
    INTERNAL = 0
    EXTERNAL = 1
    ZSYNC = 2


class DeviceSHFBase(DeviceBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reference_clock_source = ReferenceClockSourceSHF.ZSYNC
        self._is_plus: bool = False  # SHF+

    def _process_shf_opts(self):
        self._is_plus = OPT_SHF_PLUS in self.dev_opts

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

    def load_factory_preset_control_nodes(self) -> list[NodeControlBase]:
        return [
            Command(f"/{self.serial}/system/preset/load", 1),
            Response(f"/{self.serial}/system/preset/busy", 0),
            # TODO(2K): Remove once https://zhinst.atlassian.net/browse/HULK-1800 is resolved
            WaitCondition(f"/{self.serial}/system/clocks/referenceclock/in/source", 0),
            WaitCondition(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
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
            Condition(
                f"/{self.serial}/system/clocks/referenceclock/in/sourceactual", source
            ),
            Setting(f"/{self.serial}/system/clocks/referenceclock/in/source", source),
            Response(f"/{self.serial}/system/clocks/referenceclock/in/status", 0),
        ]
