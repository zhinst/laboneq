# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq import simple
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(
    types=[
        simple.AcquisitionType,
        simple.AveragingMode,
        simple.CarrierType,
        simple.ExecutionType,
        simple.CancellationSource,
        simple.RepetitionMode,
        simple.PortMode,
        simple.ModulationType,
        simple.HighPassCompensationClearing,
        simple.SectionAlignment,
    ],
    public=True,
)
class LabOneQEnumSerializer(LabOneQClassicSerializer):
    SERIALIZER_ID = "laboneq.serializers.implementations.LabOneQEnumSerializer"
    VERSION = 1
