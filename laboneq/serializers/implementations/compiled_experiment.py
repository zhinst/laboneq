# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.core.types.compiled_experiment import CompiledExperiment
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(types=CompiledExperiment, public=True)
class CompiledExperimentSerializer(LabOneQClassicSerializer[CompiledExperiment]):
    SERIALIZER_ID = "laboneq.serializers.implementations.CompiledExperimentSerializer"
    VERSION = 1
