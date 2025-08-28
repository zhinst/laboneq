# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from laboneq.core.types.enums.dsl_version import DSLVersion
from laboneq.dsl.experiment.experiment import Experiment
from laboneq.serializers._cache import create_caches
from laboneq.serializers.base import LabOneQClassicSerializer, VersionedClassSerializer
from laboneq.serializers.implementations._models._calibration import (
    remove_high_pass_clearing,
)
from laboneq.serializers.implementations._models._experiment import (
    AllSectionModel,
    ExperimentSignalModel,
    make_converter,
)
from laboneq.serializers.serializer_registry import serializer
from laboneq.serializers.types import (
    DeserializationOptions,
    JsonSerializableType,
    SerializationOptions,
)

_logger = logging.getLogger(__name__)
_converter = make_converter()


@serializer(types=Experiment, public=True)
class ExperimentSerializer(VersionedClassSerializer[Experiment]):
    SERIALIZER_ID = "laboneq.serializers.implementations.ExperimentSerializer"
    VERSION = 3

    @classmethod
    def to_dict(
        cls, obj: Experiment, options: SerializationOptions | None = None
    ) -> JsonSerializableType:
        with create_caches():
            sections = [
                _converter.unstructure(section, AllSectionModel)
                for section in obj.sections
            ]
        return {
            "__serializer__": cls.serializer_id(),
            "__version__": cls.version(),
            "__data__": {
                "uid": obj.uid,
                "name": obj.name,
                "signals": {
                    k: _converter.unstructure(v, ExperimentSignalModel)
                    for k, v in obj.signals.items()
                },
                "version": _converter.unstructure(obj.version),
                "epsilon": obj.epsilon,
                "sections": sections,
            },
        }

    @classmethod
    def from_dict_v3(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Experiment:
        serialized_data = serialized_data["__data__"]
        with create_caches():
            sections = [
                _converter.structure(section, AllSectionModel)
                for section in serialized_data["sections"]
            ]
        return Experiment(
            uid=serialized_data["uid"],
            name=serialized_data["name"],
            signals={
                k: _converter.structure(v, ExperimentSignalModel)
                for k, v in serialized_data["signals"].items()
            },
            version=_converter.structure(serialized_data["version"], DSLVersion),
            epsilon=serialized_data["epsilon"],
            sections=sections,
        )

    @classmethod
    def _replace_acquire_loop_nt_with_sweep_v2(
        cls, section_data: dict, ctx: dict | None = None
    ):
        if ctx is None:
            ctx = {"acquire_loop_nt_count": 0}
        for section in section_data:
            if section["_type"] == "AcquireLoopNt":
                _logger.warning(
                    f"Converting AcquireLoopNt {section['uid']!r} from serialized experiment (version 2) to"
                    " a Sweep. The conversion drops the 'averaging_mode' from the AcquireLoopNt."
                    " AcquireLoopNt was removed from LabOne Q in version 2.57.0."
                )
                section["_type"] = "Sweep"
                del section["averaging_mode"]
                count = section.pop("count")
                if count is None:
                    # Default count to 1. LinearSweepParameter rejects None as a count
                    # or stop value.
                    count = 1
                section["parameters"] = [
                    {
                        "_type": "LinearSweepParameter",
                        "uid": f"acquire_loop_nt_par_{ctx['acquire_loop_nt_count']}",
                        "start": 0,
                        "stop": count - 1,
                        "count": count,
                    }
                ]
                ctx["acquire_loop_nt_count"] += 1
            if isinstance(children := section.get("children"), list):
                cls._replace_acquire_loop_nt_with_sweep_v2(children, ctx=ctx)

    @classmethod
    def _remove_high_pass_clearing_v2(cls, signal_data: dict):
        for signal_uid, signal_info in signal_data.items():
            calibration_info = signal_info.get("calibration")
            remove_high_pass_clearing(signal_uid, calibration_info, _logger)

    @classmethod
    def from_dict_v2(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ):
        se = serialized_data["__data__"]
        cls._replace_acquire_loop_nt_with_sweep_v2(se["sections"])
        cls._remove_high_pass_clearing_v2(se["signals"])
        return cls.from_dict_v3(serialized_data, options)

    @classmethod
    def from_dict_v1(
        cls,
        serialized_data: JsonSerializableType,
        options: DeserializationOptions | None = None,
    ) -> Experiment:
        return LabOneQClassicSerializer.from_dict_v1(serialized_data, options)
