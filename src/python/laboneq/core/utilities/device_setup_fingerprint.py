# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""The input of `laboneq._rust.utils.device_setup_fingerprint`.

The name of `device_type` selects the device type, and an unknown name is an
error. `device_options` is the device type and its options in one
slash-separated string. `is_qc` marks one half of a split SHFQC.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FingerprintDeviceType:
    name: str


@dataclass
class FingerprintDevice:
    uid: str
    device_type: FingerprintDeviceType
    device_options: str | None
    is_qc: bool


@dataclass
class FingerprintSetup:
    devices: list[FingerprintDevice]
