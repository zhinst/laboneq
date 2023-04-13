# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import Iterator, Set, Tuple

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

    def resolve_ls_path_outputs(self, ls_path: str) -> Tuple[str, Set[int]]:
        device_uid: str = None
        outputs: Set[int] = set()
        for instrument in self._device_setup.instruments:
            for conn in instrument.connections:
                if conn.remote_path == ls_path:
                    if device_uid is None:
                        device_uid = instrument.uid
                    output_port = instrument.output_by_uid(conn.local_port)
                    dev_outputs = (
                        []
                        if output_port is None or output_port.physical_port_ids is None
                        else output_port.physical_port_ids
                    )
                    outputs.update([int(o) for o in dev_outputs])
            if device_uid is not None:
                # ignore the never-should-happen case when ls is mapped to multiple devices
                break
        return device_uid, outputs

    def get_device_used_outputs(self, device_uid: str) -> Set[int]:
        used_outputs: Set[int] = set()
        for instrument in self._device_setup.instruments:
            if instrument.uid == device_uid:
                for conn in instrument.connections:
                    output_port = instrument.output_by_uid(conn.local_port)
                    dev_outputs = (
                        []
                        if output_port is None or output_port.physical_port_ids is None
                        else output_port.physical_port_ids
                    )
                    used_outputs.update([int(o) for o in dev_outputs])
                break
        return used_outputs
