# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
from collections.abc import Mapping
from unittest.mock import MagicMock

import numpy as np
import zhinst.core
from zhinst.toolkit.driver.devices import DeviceType
from zhinst.toolkit import Session as TKSession

from laboneq.controller.devices.device_zi import DeviceBase, DeviceZI


class MockedToolkit(MagicMock):
    def __float__(self):
        return 0

    def __array__(self):
        return np.array(0)


class ToolkitDevices(Mapping):
    """Mapping for the Toolkit devices in the system.

    To learn more about how Zhinst Toolkit devices works, please refer to the
    package documentation: https://docs.zhinst.com/zhinst-toolkit/en/latest/

    Args:
        devices: Mapping of devices in the device setup.
    """

    def __init__(self, devices: dict[str, DeviceZI] | None = None):
        self._devices = (
            {u: d for u, d in devices.items() if isinstance(d, DeviceBase)}
            if devices
            else {}
        )
        self._tk_sessions: dict[tuple[str, int], TKSession] = {}

    def _tk_session(
        self, host: str, port: int, daq: zhinst.core.ziDAQServer | None
    ) -> TKSession:
        """Toolkit session from the initialized DAQ session."""
        tk_session = self._tk_sessions.get((host, port))
        if tk_session is None:
            tk_session = TKSession(server_host=host, server_port=port, connection=daq)
            self._tk_sessions[(host, port)] = tk_session
        return tk_session

    def __getitem__(self, key: str) -> DeviceType:
        """Get item.

        Both device serial (DEV1234) and instrument UID in device setup descriptor are
        recognized. Instrument UID takes precedence and is preferred.
        """
        device = self._devices.get(key)
        if device is None:
            device = next(
                (d for d in self._devices.values() if d.serial == key.lower()), None
            )

        if device is None:
            raise KeyError(f"No device found with serial or uid '{key}'")

        tk_session = self._tk_session(
            device.server_qualifier.host,
            device.server_qualifier.port,
            getattr(
                device, "_zi_api_object", None
            ),  # TODO(2K): Tests still provide a mock api object
        )

        return tk_session.devices[device.serial]

    def __iter__(self):
        return iter(self._devices)

    def __len__(self):
        return len(self._devices)
