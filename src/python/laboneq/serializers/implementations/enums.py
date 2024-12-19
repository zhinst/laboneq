# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl import enums
from laboneq.data import calibration
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(
    types=[
        enums.AcquisitionType,
        enums.AveragingMode,
        enums.CarrierType,
        enums.ExecutionType,
        calibration.CancellationSource,
        enums.RepetitionMode,
        enums.PortMode,
        enums.ModulationType,
        enums.HighPassCompensationClearing,
        enums.SectionAlignment,
    ],
    public=True,
)
class LabOneQEnumSerializer(LabOneQClassicSerializer):
    SERIALIZER_ID = "laboneq.serializers.implementations.LabOneQEnumSerializer"
    VERSION = 1
