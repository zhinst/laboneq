# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.device.device_setup import DeviceSetup
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(types=DeviceSetup, public=True)
class DeviceSetupSerializer(LabOneQClassicSerializer[DeviceSetup]):
    SERIALIZER_ID = "laboneq.serializers.implementations.DeviceSetupSerializer"
    VERSION = 1
