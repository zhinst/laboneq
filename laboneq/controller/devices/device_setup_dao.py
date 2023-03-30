# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import Iterator, Tuple

from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.device.instruments.shfqa import SHFQA
from laboneq.dsl.device.instruments.shfsg import SHFSG
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.servers.data_server import DataServer


class DeviceSetupDAO:
    def __init__(self, device_setup: DeviceSetup):
        self._device_setup = device_setup

    @property
    def instruments(self) -> Iterator[ZIStandardInstrument]:
        for instrument in self._device_setup.instruments:
            if isinstance(instrument, ZIStandardInstrument):
                yield instrument

    @property
    def servers(self) -> Iterator[Tuple[str, DataServer]]:
        return self._device_setup.servers.items()

    @cached_property
    def has_shf(self):
        for instrument in self._device_setup.instruments:
            if isinstance(instrument, (SHFQA, SHFSG)):
                return True
        return False
