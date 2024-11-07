# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from laboneq.dsl.experiment.experiment import Experiment
from laboneq.serializers.base import LabOneQClassicSerializer
from laboneq.serializers.serializer_registry import serializer


@serializer(types=Experiment, public=True)
class ExperimentSerializer(LabOneQClassicSerializer[Experiment]):
    SERIALIZER_ID = "laboneq.serializers.implementations.ExperimentSerializer"
    VERSION = 1
