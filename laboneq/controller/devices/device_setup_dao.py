# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import math
from functools import cached_property
from typing import Iterator

from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.dsl.device.instruments.shfqa import SHFQA
from laboneq.dsl.device.instruments.shfsg import SHFSG
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.servers.data_server import DataServer

_logger = logging.getLogger(__name__)


def _port_outputs(instrument: ZIStandardInstrument, local_port_path: str) -> list[int]:
    output_port = instrument.output_by_uid(local_port_path)
    dev_outputs = (
        []
        if output_port is None or output_port.physical_port_ids is None
        else output_port.physical_port_ids
    )
    return [int(o) for o in dev_outputs]


class DeviceSetupDAO:
    def __init__(self, device_setup: DeviceSetup):
        self._device_setup = device_setup

    @property
    def instruments(self) -> Iterator[ZIStandardInstrument]:
        for instrument in self._device_setup.instruments:
            if isinstance(instrument, ZIStandardInstrument):
                yield instrument

    @property
    def servers(self) -> Iterator[tuple[str, DataServer]]:
        return self._device_setup.servers.items()

    @cached_property
    def has_shf(self):
        for instrument in self._device_setup.instruments:
            if isinstance(instrument, (SHFQA, SHFSG)):
                return True
        return False

    def resolve_ls_path_outputs(self, ls_path: str) -> tuple[str, set[int]]:
        device_uid: str = None
        outputs: set[int] = set()
        for instrument in self._device_setup.instruments:
            for conn in instrument.connections:
                if conn.remote_path == ls_path:
                    if device_uid is None:
                        device_uid = instrument.uid
                    outputs.update(_port_outputs(instrument, conn.local_port))
            if device_uid is not None:
                # ignore the never-should-happen case when ls is mapped to multiple devices
                break
        return device_uid, outputs

    def get_device_used_outputs(self, device_uid: str) -> set[int]:
        used_outputs: set[int] = set()
        instrument = self._device_setup.instrument_by_uid(device_uid)
        if instrument is not None:
            for conn in instrument.connections:
                used_outputs.update(_port_outputs(instrument, conn.local_port))
        return used_outputs

    def get_device_rf_voltage_offsets(self, device_uid: str) -> dict[int, float]:
        "Returns map: <sigout index> -> <voltage_offset>"
        voltage_offsets: dict[int, float] = {}

        def add_voltage_offset(sigout: int, voltage_offset: float):
            if sigout in voltage_offsets:
                if not math.isclose(voltage_offsets[sigout], voltage_offset):
                    _logger.warning(
                        "Ambiguous 'voltage_offset' for the output %s of device %s: %s != %s, "
                        "will use %s",
                        sigout,
                        device_uid,
                        voltage_offsets[sigout],
                        voltage_offset,
                        voltage_offsets[sigout],
                    )
            else:
                voltage_offsets[sigout] = voltage_offset

        instrument = self._device_setup.instrument_by_uid(device_uid)
        if instrument is not None:
            for conn in instrument.connections:
                if conn.signal_type == IOSignalType.RF:
                    calib = self._device_setup._get_calibration(conn.remote_path)
                    if calib is not None and calib.voltage_offset is not None:
                        outputs = _port_outputs(instrument, conn.local_port)
                        if len(outputs) == 1:
                            add_voltage_offset(int(outputs[0]), calib.voltage_offset)
        return voltage_offsets
