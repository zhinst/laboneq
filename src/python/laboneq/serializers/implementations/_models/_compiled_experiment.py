# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Models for the compiled experiment."""

from __future__ import annotations

import sys
from typing import Any, Callable, ClassVar, Type

import attrs
import numpy

from laboneq.data.scheduled_experiment import (
    CompilerArtifact,
    HandleResultShape,
    ResultShapeInfo,
    RtLoopProperties,
    ScheduledExperiment,
    SoftwareVersions,
)
from laboneq.executor.executor import Statement
from laboneq.serializers._legacy.serializer import Serializer

from ._common import (
    _structure_arraylike,
    _unstructure_arraylike,
    collect_models,
    make_laboneq_converter,
    register_models,
)
from ._experiment import AcquisitionTypeModel, AveragingModeModel


@attrs.define
class RtLoopPropertiesModel:
    acquisition_type: AcquisitionTypeModel
    averaging_mode: AveragingModeModel
    shots: int
    chunk_count: int | None

    _target_class: ClassVar[Type] = RtLoopProperties


@attrs.define
class SoftwareVersionsModel:
    laboneq: str
    _target_class: ClassVar[Type] = SoftwareVersions


@attrs.define
class HandleResultShapeModel:
    shape: tuple[int, ...]
    axis_names: list[str | list[str]]
    axis_values: list[numpy.ndarray | list[numpy.ndarray]]
    chunked_axis_index: int | None
    match_case_mask: dict[int, list[int]] | None

    _target_class: ClassVar[Type] = HandleResultShape


@attrs.define
class ResultShapeInfoModel:
    shapes: dict[str, HandleResultShapeModel]

    _target_class: ClassVar[Type] = ResultShapeInfo


# Plugin registry for CompilerArtifact subclasses
# Maps artifact_type -> (target_class, unstructure_fn, structure_fn)
_compiler_artifact_plugins: dict[str, tuple[type, Callable, Callable]] = {}


@attrs.define
class CompilerArtifactModel:
    """Polymorphic model for CompilerArtifact using plugin registry."""

    _target_class: ClassVar[Type] = None

    @classmethod
    def _unstructure(cls, obj: CompilerArtifact):
        for artifact_type, (
            target_class,
            unstructure_fn,
            _,
        ) in _compiler_artifact_plugins.items():
            if isinstance(obj, target_class):
                result = unstructure_fn(obj)
                result["_artifact_type"] = artifact_type
                return result

        raise ValueError(
            f"No serialization plugin registered for {type(obj).__name__}. "
            f"Registered types: {', '.join(_compiler_artifact_plugins)}. "
            "Is the backend that produced these artifacts installed and imported?"
        )

    @classmethod
    def _structure(cls, obj: dict[str, Any], _) -> CompilerArtifact:
        artifact_type = obj.pop("_artifact_type", None)

        if artifact_type in _compiler_artifact_plugins:
            _, _, structure_fn = _compiler_artifact_plugins[artifact_type]
            return structure_fn(obj)

        raise ValueError(
            f"Unknown _artifact_type: {artifact_type}. "
            f"Available types: {', '.join(_compiler_artifact_plugins)}"
        )


@attrs.define
class ScheduledExperimentModel:
    device_setup_fingerprint: str
    rt_loop_properties: RtLoopPropertiesModel
    result_shape_info: ResultShapeInfoModel

    # NOTE! The data structure for the following fields is not completely
    # defined in the original code. We will resort to using the old serializer
    # for now (see the function make_converter below).
    # TODO: Revisit this later to swap out the old serializer
    # with the new one completely.
    artifacts: CompilerArtifactModel
    schedule: dict[str, Any] | None
    execution: Statement
    total_execution_time: float
    max_step_execution_time: float
    versions: SoftwareVersionsModel
    coprocessor_payloads: dict[str, bytes] = attrs.field(factory=dict)
    coprocessor_mappings: dict[str, str] = attrs.field(factory=dict)

    _target_class: ClassVar[Type] = ScheduledExperiment


def _old_serialize(obj):
    return Serializer.to_dict(obj)


def _old_deserialize(obj, obj_type):
    return Serializer.load(obj, obj_type)


def _unstructure_np_or_list_np(obj: numpy.ndarray | list[numpy.ndarray]):
    if isinstance(obj, list):
        return [_unstructure_arraylike(item) for item in obj]
    return _unstructure_arraylike(obj)


def _structure_np_or_list_np(obj, _) -> numpy.ndarray | list[numpy.ndarray]:
    if isinstance(obj, list):
        return [_structure_arraylike(item, _) for item in obj]
    return _structure_arraylike(obj, _)


def make_converter():
    converter = make_laboneq_converter()

    # NOTE! For some objects we still resort to the old serializer for now, for two reasons:
    # 1. The data structure for some fields is not completely defined in the original code.
    # 2. To cut some corners during the new serializer implementation.
    # Hence, we manually register custom hooks for them.
    # We have to register these custom hooks first. Otherwise, the automatic hooks for parent attrs classes
    # silently assume some default repr based implementation, which does not get overridden if custom hooks
    # are registered later. The automatic hooks are generated via make_dict_unstructure_fn /
    # make_dict_structure_fn.
    # Note: CompilerArtifact is now handled via the plugin registry in CompilerArtifactModel
    for cls in [Statement, dict[str, Any] | None]:
        converter.register_unstructure_hook(cls, _old_serialize)
        converter.register_structure_hook(cls, _old_deserialize)

    # The type of HandleResultShapeModel.axis_values is simple, yet the serializer is not able to consume it,
    # even though we have serializers for both list and numpy.ndarray. Thus, have to register special hooks.
    converter.register_unstructure_hook(
        numpy.ndarray | list[numpy.ndarray], _unstructure_np_or_list_np
    )
    converter.register_structure_hook(
        numpy.ndarray | list[numpy.ndarray], _structure_np_or_list_np
    )

    register_models(converter, collect_models(sys.modules[__name__]))
    return converter


def register_compiler_artifact_plugin(
    artifact_type: str,
    target_class: type,
    unstructure_fn: Callable,
    structure_fn: Callable,
) -> None:
    """
    Register a CompilerArtifact plugin for optional artifact types.

    Args:
        artifact_type: The discriminator value (e.g., "QCCS")
        target_class: The target class (e.g., ArtifactsCodegen)
        unstructure_fn: Function to serialize the artifact to a dict
        structure_fn: Function to deserialize a dict to the artifact
    """
    _compiler_artifact_plugins[artifact_type] = (
        target_class,
        unstructure_fn,
        structure_fn,
    )
