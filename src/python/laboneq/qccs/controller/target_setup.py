# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Conversion of a DSL device setup into the QCCS target setup."""

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.qccs.controller.target_setup_generator import (
    TargetSetupGenerator,
)
from laboneq.qccs.setup_converter import (
    convert_device_setup_to_setup,
)

if TYPE_CHECKING:
    from laboneq.data.execution_payload import TargetSetup
    from laboneq.dsl.device.device_setup import DeviceSetup


def build(device_setup: DeviceSetup) -> TargetSetup:
    """The QCCS target setup for `device_setup`."""
    return TargetSetupGenerator.from_setup(convert_device_setup_to_setup(device_setup))
