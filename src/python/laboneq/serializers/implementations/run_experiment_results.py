# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from laboneq.dsl.result import Results, AcquiredResult
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.implementations.results import _acquired_axis_to_ndarrays
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)


class RunExperimentResults:
    """Removed RunExperimentResults class.

    The RunExperimentResults class no longer exists. This
    class exists only as a marker for registering the
    deserializer for data saved when it existed.
    """


@serializer(types=RunExperimentResults, public=True)
class RunExperimentResultsSerializer(VersionedClassSerializer[Results]):
    SERIALIZER_ID = "laboneq.serializers.implementations.RunExperimentResultsSerializer"
    VERSION = 1

    @classmethod
    def to_dict(
        cls, obj: RunExperimentResults, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        raise TypeError(
            "The RunExperimentResults class no longer exists and thus cannot be serialized."
            " Use the Results serializer instead."
        )

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Results:
        data = serialized_data["__data__"]
        acquired_data = {
            k: AcquiredResult(
                handle=k,
                data=np.array(v["data.real"]) + 1j * np.array(v["data.imag"]),
                axis_name=v["axis_name"],
                axis=_acquired_axis_to_ndarrays(v["axis"]),
            )
            for k, v in data["acquired_data"].items()
        }
        return Results(
            acquired_results=acquired_data,
            neartime_callback_results=data["neartime_callbacks"],
            execution_errors=data["errors"],
        )
