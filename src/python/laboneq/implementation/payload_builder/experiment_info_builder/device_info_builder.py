# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
from laboneq.data.compilation_job import (
    DeviceInfo,
    DeviceInfoType,
    ReferenceClockSourceInfo,
)
from laboneq.data.setup_description import (
    DeviceType,
)

if TYPE_CHECKING:
    from laboneq.data.setup_description import (
        Instrument,
        LogicalSignal,
        Setup,
    )


def _ref_clk_from_ds(
    ref_clk: ReferenceClockSource | None,
) -> ReferenceClockSourceInfo | None:
    if ref_clk == ReferenceClockSource.INTERNAL:
        return ReferenceClockSourceInfo.INTERNAL
    elif ref_clk == ReferenceClockSource.EXTERNAL:
        return ReferenceClockSourceInfo.EXTERNAL
    return None


def _build_device_info(device: Instrument) -> DeviceInfo:
    return DeviceInfo(
        uid=device.uid,
        device_type=DeviceInfoType(device.device_type.name.lower()),
        options=device.device_options or "",
        reference_clock_source=_ref_clk_from_ds(device.reference_clock.source),
    )


class DeviceInfoBuilder:
    """Make `DeviceInfo` for each instrument defined in `Setup`."""

    def __init__(self, setup: Setup):
        self._setup = setup
        self._device_mapping: dict[str, DeviceInfo] = {}
        self._device_by_ls: dict[LogicalSignal, DeviceInfo] = {}
        self._build_devices()

    @property
    def device_mapping(self) -> Mapping[str, DeviceInfo]:
        return self._device_mapping

    def device_by_ls(self, ls: LogicalSignal) -> DeviceInfo:
        return self._device_by_ls[ls]

    def _build_devices(self):
        for device in self._setup.instruments:
            if device.device_type == DeviceType.UNMANAGED:
                continue
            device_info = _build_device_info(device)
            self._device_mapping[device.uid] = device_info
            for conn in device.connections:
                self._device_by_ls[conn.logical_signal] = self._device_mapping[
                    device.uid
                ]
