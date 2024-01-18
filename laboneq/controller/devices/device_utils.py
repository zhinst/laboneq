# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from laboneq.controller.devices.device_zi import DeviceQualifier, DeviceZI


def calc_dev_type(device_qualifier: DeviceQualifier) -> str:
    if device_qualifier.options.is_qc is True:
        return "SHFQC"
    else:
        return device_qualifier.driver


def dev_api(device: DeviceZI) -> tuple[Any, str]:
    """Temporary helper to unify emulation interface for the async API."""
    return (device._api or device._daq._zi_api_object, device.serial)
