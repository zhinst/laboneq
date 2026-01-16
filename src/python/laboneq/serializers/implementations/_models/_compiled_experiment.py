# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Models for the compiled experiment."""

from __future__ import annotations

import re
import sys
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Type, Union

import attrs
import numpy

from laboneq.core.types.enums.awg_signal_type import AWGSignalType
from laboneq.data.awg_info import AwgKey
from laboneq.data.recipe import (
    AWG,
    IO,
    AcquireLength,
    Config,
    Gains,
    Initialization,
    IntegratorAllocation,
    Measurement,
    NtStepKey,
    OscillatorParam,
    RealtimeExecutionInit,
    Recipe,
    RefClkType,
    RoutedOutput,
    SoftwareVersions,
    TriggeringMode,
)
from laboneq.data.scheduled_experiment import (
    CompilerArtifact,
    HandleResultShape,
    ResultShapeInfo,
    ResultSource,
    RtLoopProperties,
    ScheduledExperiment,
)
from laboneq.dsl.serialization import Serializer
from laboneq.executor.executor import Statement

from ._common import (
    _structure_arraylike,
    _unstructure_arraylike,
    collect_models,
    make_laboneq_converter,
    register_models,
)
from ._experiment import AcquisitionTypeModel, AveragingModeModel

# Models for Recipe:


class AWGSignalTypeModel(Enum):
    IQ = "iq"
    SINGLE = "single"
    DOUBLE = "double"
    _target_class = AWGSignalType


class RefClkTypeModel(Enum):
    _10MHZ = 10_000_000
    _100MHZ = 100_000_000
    _target_class = RefClkType


class TriggeringModeModel(Enum):
    ZSYNC_FOLLOWER = 1
    DIO_FOLLOWER = 2
    DESKTOP_LEADER = 3
    DESKTOP_DIO_FOLLOWER = 4
    INTERNAL_FOLLOWER = 5
    _target_class = TriggeringMode


@attrs.define
class NtStepKeyModel:
    indices: tuple[int, ...]
    _target_class: ClassVar[Type] = NtStepKey


@attrs.define
class GainsModel:
    diagonal: Union[float, str]
    off_diagonal: Union[float, str]
    _target_class: ClassVar[Type] = Gains


@attrs.define
class RoutedOutputModel:
    from_channel: int
    amplitude: Union[float, str]
    phase: Union[float, str]
    _target_class: ClassVar[Type] = RoutedOutput


@attrs.define
class IOModel:
    # type of following attributes are not clear from the original code.
    # precompensation, lo_frequency, port_delay, and amplitude
    # Use Any for cattrs to pass it through
    channel: int
    enable: bool | None
    modulation: bool | None
    offset: float | str | None
    gains: GainsModel | None
    range: float | None
    range_unit: str | None
    precompensation: Any
    lo_frequency: Any
    port_mode: str | None
    port_delay: Any
    scheduler_port_delay: float
    delay_signal: float | None
    marker_mode: str | None
    amplitude: Any
    routed_outputs: list[RoutedOutputModel]
    enable_output_mute: bool
    _target_class: ClassVar[Type] = IO


@attrs.define
class AWGModel:
    awg: int | str
    signal_type: AWGSignalTypeModel
    signals: dict[str, dict[str, str]]
    source_feedback_register: int | Literal["local"] | None
    codeword_bitshift: int | None
    codeword_bitmask: int | None
    feedback_register_index_select: int | None
    command_table_match_offset: int | None
    target_feedback_register: int | None
    _target_class: ClassVar[Type] = AWG


@attrs.define
class MeasurementModel:
    length: int
    channel: int = 0
    _target_class: ClassVar[Type] = Measurement


@attrs.define
class ConfigModel:
    repetitions: int
    triggering_mode: TriggeringModeModel
    sampling_rate: float | None
    _target_class: ClassVar[Type] = Config


@attrs.define
class InitializationModel:
    device_uid: str
    device_type: str | None
    config: ConfigModel
    awgs: list[AWGModel]
    outputs: list[IOModel]
    inputs: list[IOModel]
    measurements: list[MeasurementModel]

    # assume ppchannels is a list of dictionaries with simple values
    ppchannels: list[dict[str, str | int | float | bool]]
    _target_class: ClassVar[Type] = Initialization


@attrs.define
class OscillatorParamModel:
    id: str
    device_id: str
    channel: int
    signal_id: str
    allocated_index: int
    frequency: Optional[float]
    param: Optional[str]
    _target_class: ClassVar[Type] = OscillatorParam


@attrs.define
class IntegratorAllocationModel:
    signal_id: str
    device_id: str
    awg: int
    channels: list[int]
    kernel_count: int
    thresholds: list[float]
    _target_class: ClassVar[Type] = IntegratorAllocation


@attrs.define
class AcquireLengthModel:
    signal_id: str
    acquire_length: int
    _target_class: ClassVar[Type] = AcquireLength


@attrs.define
class RealtimeExecutionInitModel:
    device_id: str
    awg_index: int
    program_ref: str
    nt_step: NtStepKeyModel
    wave_indices_ref: str | None
    kernel_indices_ref: str | None
    _target_class: ClassVar[Type] = RealtimeExecutionInit


@attrs.define
class RtLoopPropertiesModel:
    uid: str
    acquisition_type: AcquisitionTypeModel
    averaging_mode: AveragingModeModel
    shots: int
    chunk_count: int | None

    _target_class: ClassVar[Type] = RtLoopProperties


@attrs.define
class SoftwareVersionsModel:
    target_labone: str
    laboneq: str
    _target_class: ClassVar[Type] = SoftwareVersions


@attrs.define
class HandleResultShapeModel:
    signal: str
    shape: tuple[int, ...]
    axis_names: list[str | list[str]]
    axis_values: list[numpy.ndarray | list[numpy.ndarray]]
    chunked_axis_index: int | None
    match_case_mask: dict[int, list[int]] | None

    _target_class: ClassVar[Type] = HandleResultShape


@attrs.define
class ResultShapeInfoModel:
    shapes: dict[str, HandleResultShapeModel]
    result_handle_maps: dict[ResultSource, list[set[str]]]
    result_lengths: dict[AwgKey, int]

    _target_class: ClassVar[Type] = ResultShapeInfo


@attrs.define
class RecipeModel:
    initializations: list[InitializationModel]
    realtime_execution_init: list[RealtimeExecutionInitModel]
    oscillator_params: list[OscillatorParamModel]
    integrator_allocations: list[IntegratorAllocationModel]
    acquire_lengths: list[AcquireLengthModel]
    total_execution_time: float
    max_step_execution_time: float
    is_spectroscopy: bool
    versions: SoftwareVersionsModel
    _target_class: ClassVar[Type] = Recipe


@attrs.define
class ScheduledExperimentModel:
    device_setup_fingerprint: str
    recipe: RecipeModel
    rt_loop_properties: RtLoopPropertiesModel
    result_shape_info: ResultShapeInfoModel

    # NOTE! The data structure for the following fields is not completely
    # defined in the original code. We will resort to using the old serializer
    # for now (see the function make_converter below).
    # TODO: Revisit this later to swap out the old serializer
    # with the new one completely.
    artifacts: CompilerArtifact
    schedule: dict[str, Any] | None
    execution: Statement

    _target_class: ClassVar[Type] = ScheduledExperiment


def _old_serialize(obj):
    return Serializer.to_dict(obj)


def _old_deserialize(obj, obj_type):
    return Serializer.load(obj, obj_type)


def _unstructure_awg_key(obj: AwgKey):
    return f"AwgKey({obj.device_id}, {obj.awg_id})"


def _structure_awg_key(obj, _) -> AwgKey:
    match_result = re.fullmatch(r"AwgKey\((.*), (.*)\)", obj)
    assert match_result is not None
    device_id, awg_idx = match_result.groups()
    if awg_idx.isnumeric():
        awg_idx = int(awg_idx)
    return AwgKey(device_id, awg_idx)


def _unstructure_result_source(obj: ResultSource):
    return f"ResultSource({obj.device_id}, {obj.awg_id}, {obj.integrator_idx})"


def _structure_result_source(obj, _) -> ResultSource:
    match_result = re.fullmatch(r"ResultSource\((.*), (.*), (.*)\)", obj)
    assert match_result is not None
    device_id, awg_id, integrator_idx = match_result.groups()
    if awg_id.isnumeric():
        awg_id = int(awg_id)
    return ResultSource(device_id, awg_id, int(integrator_idx))


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

    # NOTE! Because of 1. The data structure for some fields is not completely defined in the original code,
    # 2. To cut some corners during the new serializer implementation; for some objects we still resort to
    # the old serializer for now. Hence, we manually register custom hooks for them.
    # We have to register these custom hooks first, otherwise the automatic hooks generated for parent attrs
    # classes (via make_dict_unstructure_fn / make_dict_structure_fn) silently assume some default repr based
    # implementation which does not get overridden if custom hooks are registered later.
    for cls in [Statement, CompilerArtifact, dict[str, Any] | None]:
        converter.register_unstructure_hook(cls, _old_serialize)
        converter.register_structure_hook(cls, _old_deserialize)

    # For AwgKey and ResultSource we register custom serializer/deserializer, since they are used as dictionary key
    # and cannot be serialized to a dict
    converter.register_unstructure_hook(AwgKey, _unstructure_awg_key)
    converter.register_structure_hook(AwgKey, _structure_awg_key)
    converter.register_unstructure_hook(ResultSource, _unstructure_result_source)
    converter.register_structure_hook(ResultSource, _structure_result_source)

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
