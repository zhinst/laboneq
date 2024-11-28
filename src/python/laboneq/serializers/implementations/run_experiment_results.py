# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from laboneq.workflow.tasks import RunExperimentResults
from laboneq.workflow.tasks.run_experiment import AcquiredResult
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    SerializationOptions,
    DeserializationOptions,
    JsonSerializableType,
)


@serializer(types=RunExperimentResults, public=True)
class RunExperimentResultsSerializer(VersionedClassSerializer[RunExperimentResults]):
    SERIALIZER_ID = "laboneq.serializers.implementations.RunExperimentResultsSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: RunExperimentResults, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "acquired_data": {
                    k: {
                        "data.real": np.ascontiguousarray(np.real(v.data)),
                        "data.imag": np.ascontiguousarray(np.imag(v.data)),
                        "axis_name": v.axis_name,
                        "axis": v.axis,
                    }
                    for k, v in obj._data.items()
                },
                "neartime_callbacks": obj._neartime_callbacks,
                "errors": obj._errors,
            },
        }

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> RunExperimentResults:
        data = serialized_data["__data__"]
        acquired_data = {
            k: AcquiredResult(
                data=np.array(v["data.real"]) + 1j * np.array(v["data.imag"]),
                axis_name=v["axis_name"],
                axis=v["axis"],
            )
            for k, v in data["acquired_data"].items()
        }
        return RunExperimentResults(
            data=acquired_data,
            neartime_callbacks=data["neartime_callbacks"],
            errors=data["errors"],
        )
