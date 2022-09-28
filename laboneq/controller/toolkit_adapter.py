# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional
from collections.abc import Mapping

from zhinst.toolkit.driver.devices import DeviceType
from laboneq.controller.devices.device_zi import DeviceZI


class ToolkitDevices(Mapping):
    """Mapping for the Toolkit devices in the system.

    To learn more about how Zhinst Toolkit devices works, please refer to the
    package documentation: https://docs.zhinst.com/zhinst-toolkit/en/latest/

    Args:
        devices: Mapping of devices in the device setup.
    """

    def __init__(self, devices: Optional[Dict[str, DeviceZI]] = None):
        self._devices = devices if devices else {}

    def __getitem__(self, key) -> DeviceType:
        """Get item.

        Both device serial (DEV1234) and instrument UID in device setup descriptor are
        recognized. Instrument UID takes precedence and is preferred.
        """
        try:
            device = self._devices[key]
            return device.daq.toolkit_session.devices[device.serial]
        except KeyError as error:
            for device in self._devices.values():
                if device.serial.lower() == key.lower():
                    return device.daq.toolkit_session.devices[key]
            raise error

    def __iter__(self):
        return iter(self._devices)

    def __len__(self):
        return len(self._devices)
