# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import pybase64

from laboneq import __version__
from laboneq.core.types.compiled_experiment import (
    CompiledExperiment,
    compiler_output_bytes,
)
from laboneq.data.scheduled_experiment import ScheduledExperiment
from laboneq.serializers.base import VersionedClassSerializer
from laboneq.serializers.core import from_dict, to_dict
from laboneq.serializers.implementations._models._compiled_experiment import (
    ScheduledExperimentModel,
    make_converter,
)
from laboneq.serializers.serializer_registry import serializer

if TYPE_CHECKING:
    from laboneq.serializers.types import (
        DeserializationOptions,
        JsonSerializableType,
        SerializationOptions,
    )

_converter = make_converter()


def _check_laboneq_version(
    serialized_laboneq_version: str | None,
    options: DeserializationOptions | None = None,
) -> None:
    """Check serialized data LabOne Q version.

    Rejects data written by a different LabOne Q version, unless `force` is set in the options.
    """
    check_version = options is None or not options.force
    if serialized_laboneq_version is None:
        NOT_FOUND_MSG = "Could not find LabOne Q version in serialized data."
        if check_version:
            raise ValueError(NOT_FOUND_MSG)
        else:
            warnings.warn(NOT_FOUND_MSG, UserWarning, stacklevel=2)
    elif serialized_laboneq_version != __version__:
        MISMATCH_MSG = (
            f"LabOne Q version mismatch. Check out the Labone Q with correct version "
            f"{serialized_laboneq_version} to load the serialized data. Otherwise, set "
            f"the `force` option to True to skip the version check."
        )
        if check_version:
            raise ValueError(MISMATCH_MSG)
        else:
            warnings.warn(MISMATCH_MSG, UserWarning, stacklevel=2)


def _unstructure_payloads(payloads: dict[str, bytes]) -> dict[str, str]:
    """Base64-encode coprocessor payloads to a JSON-safe dict of ascii strings."""
    return {
        key: pybase64.b64encode(payload).decode("ascii")
        for key, payload in payloads.items()
    }


def _structure_payloads(d: dict[str, str]) -> dict[str, bytes]:
    """Base64-decode coprocessor payloads from a JSON-safe dict."""
    return {key: pybase64.b64decode(value.encode("ascii")) for key, value in d.items()}


@serializer(types=CompiledExperiment, public=True)
class CompiledExperimentSerializer(VersionedClassSerializer[CompiledExperiment]):
    SERIALIZER_ID = "laboneq.serializers.implementations.CompiledExperimentSerializer"
    VERSION = 4

    @classmethod
    def to_dict(
        cls, obj: CompiledExperiment, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__laboneq_version__": __version__,
            "__data__": {
                "device_setup": to_dict(obj.device_setup, options),
                "experiment": to_dict(obj.experiment, options),
                "compiler_output": pybase64.b64encode(
                    compiler_output_bytes(obj)
                ).decode("ascii"),
                "coprocessor_payload_overrides": _unstructure_payloads(
                    obj.coprocessor_payload_overrides
                ),
            },
        }

    @classmethod
    def from_dict_v4(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> CompiledExperiment:
        _check_laboneq_version(serialized_data.get("__laboneq_version__"), options)

        data = serialized_data["__data__"]
        compiled = CompiledExperiment(
            device_setup=from_dict(data["device_setup"], options),
            experiment=from_dict(data["experiment"], options),
            _compiler_output=pybase64.b64decode(data["compiler_output"]),
            coprocessor_payload_overrides=_structure_payloads(
                data["coprocessor_payload_overrides"]
            ),
        )
        return compiled


@serializer(types=ScheduledExperiment, public=True)
class ScheduledExperimentSerializer(VersionedClassSerializer[ScheduledExperiment]):
    """Serializer for a bare `ScheduledExperiment`.

    Used for a `ScheduledExperiment` held independently of a `CompiledExperiment`.
    """

    SERIALIZER_ID = "laboneq.serializers.implementations.ScheduledExperimentSerializer"
    VERSION = 2

    @classmethod
    def to_dict(
        cls, obj: ScheduledExperiment, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            scheduled_experiment = _converter.unstructure(obj, ScheduledExperimentModel)
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__laboneq_version__": __version__,
            "__data__": scheduled_experiment,
        }

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> ScheduledExperiment:
        assert isinstance(serialized_data, dict)
        _check_laboneq_version(serialized_data.get("__laboneq_version__"), options)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            return _converter.structure(
                serialized_data["__data__"],
                ScheduledExperimentModel,
            )
