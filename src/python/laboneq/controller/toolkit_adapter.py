# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zhinst.toolkit import Session as TKSession
    from zhinst.toolkit.driver.devices import DeviceType

    from laboneq.dsl.device.device_setup import DeviceSetup


@dataclass(frozen=True)
class ToolkitInstrument:
    """An instrument as the toolkit reaches it: its serial and its data server."""

    serial: str
    host: str
    port: int


def toolkit_instruments(device_setup: DeviceSetup) -> dict[str, ToolkitInstrument]:
    """Returns the instruments of `device_setup` that a data server serves, by uid."""
    # Imported here: `laboneq.dsl` imports `laboneq.controller`, which imports
    # this module.
    from laboneq.dsl.device.instruments.zi_standard_instrument import (
        ZIStandardInstrument,
    )

    instruments: dict[str, ToolkitInstrument] = {}
    for instrument in device_setup.instruments:
        if not isinstance(instrument, ZIStandardInstrument):
            continue
        server = device_setup.servers.get(instrument.server_uid or "")
        if server is None or instrument.address is None:
            continue
        instruments[instrument.uid] = ToolkitInstrument(
            serial=instrument.address.lower(),
            host=server.host,
            port=int(server.port),
        )
    return instruments


class ToolkitDevices(Mapping):
    """Mapping for the Toolkit devices in the system.

    To learn more about how Zhinst Toolkit devices works, please refer to the
    package documentation: https://docs.zhinst.com/labone_api_user_manual/reference/toolkit/

    Args:
        devices: The instruments of the device setup, by uid, as
            `toolkit_instruments` returns them.
    """

    def __init__(self, devices: Mapping[str, ToolkitInstrument] | None = None):
        self._devices = dict(devices) if devices else {}
        self._tk_sessions: dict[tuple[str, int], TKSession] = {}

    def _tk_session(self, host: str, port: int) -> TKSession:
        tk_session = self._tk_sessions.get((host, port))
        if tk_session is None:
            # Imported lazily: zhinst.toolkit is a large import and is only needed
            # once a user actually reaches for a toolkit device.
            from zhinst.toolkit import Session as TKSession

            tk_session = TKSession(server_host=host, server_port=port)
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

        tk_session = self._tk_session(device.host, device.port)

        return tk_session.devices[device.serial]

    def __iter__(self):
        return iter(self._devices)

    def __len__(self):
        return len(self._devices)
