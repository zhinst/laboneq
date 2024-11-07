# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.result.results import Results
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(types=Results, public=True)
class ResultsSerializer(LabOneQClassicSerializer[Results]):
    SERIALIZER_ID = "laboneq.serializers.implementations.ResultsSerializer"
    VERSION = 1
