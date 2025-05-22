# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from laboneq.dsl.result.results import Results, AcquiredResult
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.core import from_dict, to_dict
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)

if TYPE_CHECKING:
    from numpy import typing as npt


def _acquired_axis_to_ndarrays(
    axis: list[list[Any] | list[list[Any]]],
) -> list[npt.NDArray[Any] | list[npt.NDArray[Any]]]:
    """Convert axis entries retrieved during deserialization to numpy arrays.

    The axis data is either a list of numpy arrays, or a list of lists of
    numpy arrays. In the serialized data the numpy arrays are lists.
    """
    if not axis:
        # empty axis data
        return axis

    if isinstance(axis[0], np.ndarray) or (
        axis[0] and isinstance(axis[0][0], np.ndarray)
    ):
        # list of numpy arrays, or lists of lists of numpy arrays
        # (these occur when deserializing dicts that were not read from JSON)
        return axis

    list_of_arrays = axis[0] and not isinstance(axis[0][0], list)

    if list_of_arrays:
        # list of numpy arrays
        return [np.array(x) for x in axis]
    else:
        # list of lists of numpy arrays
        return [[np.array(x) for x in item] for item in axis]


@serializer(types=Results, public=True)
class ResultsSerializer(VersionedClassSerializer[Results]):
    SERIALIZER_ID = "laboneq.serializers.implementations.ResultsSerializer"
    VERSION = 2

    @classmethod
    def _acquired_data_to_dict(
        cls, data: npt.NDArray[Any] | np.complex128
    ) -> dict[str, list[float] | float]:
        """Return a dictionary describing the acquired data."""
        if np.isscalar(data):
            data_real = np.real(data)
            data_imag = np.imag(data)
        elif isinstance(data, np.ndarray):
            data_real = np.ascontiguousarray(np.real(data))
            data_imag = np.ascontiguousarray(np.imag(data))
        else:
            raise TypeError(
                "AcquiredResult .data must be either a scalar or NumPy ndarray."
            )
        return {
            "data.real": data_real,
            "data.imag": data_imag,
        }

    @classmethod
    def _acquired_data_from_dict(
        cls, data_real: list[float] | float, data_imag: list[float] | float
    ) -> npt.NDArray[Any] | np.complex128:
        """Return the acquired data."""
        if np.isscalar(data_real) and np.isscalar(data_imag):
            return np.complex128(data_real + 1j * data_imag)
        return np.array(data_real) + 1j * np.array(data_imag)

    @classmethod
    def to_dict(
        cls, obj: Results, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        metadata = {}
        if obj.device_setup is not None:
            metadata["device_setup"] = to_dict(obj.device_setup)
        if obj.experiment is not None:
            metadata["experiment"] = to_dict(obj.experiment)
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "acquired_results": {
                    k: {
                        "handle": v.handle,
                        "axis_name": v.axis_name,
                        "axis": v.axis,
                        "last_nt_step": v.last_nt_step,
                        **cls._acquired_data_to_dict(v.data),
                    }
                    for k, v in obj.acquired_results.items()
                },
                "neartime_callback_results": obj.neartime_callback_results,
                "execution_errors": obj.execution_errors,
                "pipeline_jobs_timestamps": obj.pipeline_jobs_timestamps,
                **metadata,
            },
        }

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Results:
        data = serialized_data["__data__"]
        acquired_results = {
            k: AcquiredResult(
                handle=v["handle"],
                axis_name=v["axis_name"],
                axis=_acquired_axis_to_ndarrays(v["axis"]),
                data=cls._acquired_data_from_dict(v["data.real"], v["data.imag"]),
                last_nt_step=v["last_nt_step"],
            )
            for k, v in data["acquired_results"].items()
        }

        if "device_setup" in data:
            device_setup = from_dict(data["device_setup"])
        else:
            device_setup = None
        if "experiment" in data:
            experiment = from_dict(data["experiment"])
        else:
            experiment = None

        return Results(
            acquired_results=acquired_results,
            neartime_callback_results=data["neartime_callback_results"],
            execution_errors=data["execution_errors"],
            pipeline_jobs_timestamps=data["pipeline_jobs_timestamps"],
            device_setup=device_setup,
            experiment=experiment,
        )

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Results:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
