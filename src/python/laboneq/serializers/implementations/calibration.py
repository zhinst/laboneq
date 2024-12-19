# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.calibration.calibration import Calibration
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(types=Calibration, public=True)
class CalibrationSerializer(LabOneQClassicSerializer[Calibration]):
    SERIALIZER_ID = "laboneq.serializers.implementations.CalibrationSerializer"
    VERSION = 1
