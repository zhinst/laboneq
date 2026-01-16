# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""System Profile data structures for hardware capabilities.

A SystemProfile captures the current hardware configuration, including:
- Device capabilities (types, options, versions)
- Channel types and their capabilities

System profiles are auto-generated from live hardware and cached locally
to enable offline compilation. Defaults are provided for offline testing.
"""

import datetime
import hashlib

import attrs
import yaml

from laboneq._version import get_version

from . import SystemProfile


@attrs.define
class DeviceCapabilitiesQCCS:
    device_model: str | None = None
    device_options: list[str] = attrs.field(default=attrs.Factory(list[str]))


@attrs.define
class SystemProfileQCCS(SystemProfile):
    version: str = "1.0"
    generated_at: datetime.datetime = attrs.Factory(datetime.datetime.now)
    laboneq_version: str = attrs.field(default=get_version())

    setup_uid: str = ""
    server_address: str = ""
    server_port: int = 0
    server_version: str = ""

    devices: dict[str, DeviceCapabilitiesQCCS] = attrs.field(
        default=attrs.Factory(dict[str, DeviceCapabilitiesQCCS])
    )

    def get_fingerprint(self) -> str:
        essential = attrs.asdict(
            self,
            filter=attrs.filters.exclude(
                attrs.fields(SystemProfileQCCS).generated_at,
                attrs.fields(SystemProfileQCCS).laboneq_version,
            ),
        )
        content = yaml.dump(essential, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()

    @property
    def uid(self) -> str:
        """Unique identifier for the device setup."""
        return self.setup_uid

    @uid.setter
    def uid(self, value: str):
        self.setup_uid = value
