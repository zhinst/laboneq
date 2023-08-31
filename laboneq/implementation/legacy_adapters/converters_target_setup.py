# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.data.execution_payload import TargetSetup
from laboneq.implementation.legacy_adapters.device_setup_converter import (
    convert_device_setup_to_setup,
)
from laboneq.implementation.payload_builder.target_setup_generator import (
    TargetSetupGenerator,
)

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup


def convert_dsl_to_target_setup(device_setup: DeviceSetup) -> TargetSetup:
    new_setup = convert_device_setup_to_setup(device_setup)
    target_setup = TargetSetupGenerator.from_setup(new_setup)
    return target_setup
