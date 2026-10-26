# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import zhinst.core
    from zhinst.toolkit import Session as TKSession
    from zhinst.toolkit.driver.devices import DeviceType

    from laboneq.controller.devices.device_base import DeviceBase


class ToolkitDevices(Mapping):
    """Mapping for the Toolkit devices in the system.

    To learn more about how Zhinst Toolkit devices works, please refer to the
    package documentation: https://docs.zhinst.com/labone_api_user_manual/reference/toolkit/

    Args:
        devices: Mapping of devices in the device setup.
    """

    def __init__(self, devices: dict[str, DeviceBase] | None = None):
        self._devices = dict(devices) if devices else {}
        self._tk_sessions: dict[tuple[str, int], TKSession] = {}

    def _tk_session(
        self, host: str, port: int, daq: zhinst.core.ziDAQServer | None
    ) -> TKSession:
        """Toolkit session from the initialized DAQ session."""
        tk_session = self._tk_sessions.get((host, port))
        if tk_session is None:
            # Imported lazily: zhinst.toolkit is a large import and is only needed
            # once a user actually reaches for a toolkit device.
            from zhinst.toolkit import Session as TKSession

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
