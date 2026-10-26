# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""QCCS ArtifactsCodegen serialization models."""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from enum import Enum
from typing import Any, ClassVar, Type, Union

import attrs
import orjson

from laboneq.core.types.enums.awg_signal_type import AWGSignalType
from laboneq.core.types.enums.port_mode import PortMode
from laboneq.data.artifacts_qccs import (
    AWG,
    IO,
    AcquireLength,
    ArtifactsCodegen,
    Config,
    Gains,
    Initialization,
    IntegratorAllocation,
    Measurement,
    OscillatorParam,
    PPChannel,
    RealtimeExecutionInit,
    ResultSource,
    RoutedOutput,
)
from laboneq.data.nt_step_key import NtStepKey
from laboneq.serializers._legacy.serializer import Serializer
from laboneq.serializers.implementations._models._calibration import (
    CancellationSourceModel,
)

from ._common import collect_models, make_laboneq_converter, register_models
from ._compiled_experiment import register_compiler_artifact_plugin


class AWGSignalTypeModel(Enum):
    IQ = "iq"
    SINGLE = "single"
    DOUBLE = "double"
    _target_class: ClassVar[Type] = AWGSignalType  # type: ignore[misc]


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
    amplitude: float | str | None
    phase: float | str | None
    _target_class: ClassVar[Type] = RoutedOutput


@attrs.define
class PrecompensationModel:
    precompensation: dict | None
    _target_class: ClassVar[Type] = dict

    @classmethod
    def _unstructure(cls, obj):
        return orjson.dumps(obj).decode() if obj is not None else None

    @classmethod
    def _structure(cls, obj, _):
        return orjson.loads(obj) if obj is not None else None


@attrs.define
class IOModel:
    channel: int
    enable: bool | None
    modulation: bool | None
    offset: float | str | None
    gains: GainsModel | None
    range: float | None
    range_unit: str | None
    precompensation: PrecompensationModel | None
    lo_frequency: float | str | None
    port_mode: PortMode | None
    port_delay: float | str | None
    scheduler_port_delay: float
    marker_mode: str | None
    amplitude: float | str | None
    routed_outputs: list[RoutedOutputModel]
    enable_output_mute: bool
    _target_class: ClassVar[Type] = IO


@attrs.define
class AWGModel:
    awg: int
    signal_type: AWGSignalTypeModel
    signals: set[str]
    source_feedback_register: int | str | None
    codeword_bitshift: int | None
    codeword_bitmask: int | None
    feedback_register_index_select: int | None
    command_table_match_offset: int | None
    target_feedback_register: int | str | None
    result_length: int | None
    _target_class: ClassVar[Type] = AWG


@attrs.define
class MeasurementModel:
    length: int
    channel: int = 0
    _target_class: ClassVar[Type] = Measurement


@attrs.define
class ConfigModel:
    lead_delay: float
    sampling_rate: float | None
    _target_class: ClassVar[Type] = Config


@attrs.define
class PPChannelModel:
    channel: int
    pump_on: bool
    cancellation_on: bool
    cancellation_source: CancellationSourceModel
    cancellation_source_frequency: float | str | None
    alc_on: bool
    pump_filter_on: bool
    probe_on: bool
    pump_frequency: float | str | None
    pump_power: float | str | None
    probe_frequency: float | str | None
    probe_power: float | str | None
    cancellation_phase: float | str | None
    cancellation_attenuation: float | str | None
    sweep_config: str | None
    _target_class: ClassVar[Type] = PPChannel


@attrs.define
class InitializationModel:
    device_uid: str
    device_type: str | None
    config: ConfigModel
    awgs: list[AWGModel]
    outputs: list[IOModel]
    inputs: list[IOModel]
    measurements: list[MeasurementModel]
    ppchannels: list[PPChannelModel]
    _target_class: ClassVar[Type] = Initialization


@attrs.define
class OscillatorParamModel:
    id: str
    device_id: str
    channel: int
    signal_id: str
    allocated_index: int
    frequency: float | None
    param: str | None
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
class ResultSourceModel:
    device_id: str
    awg_id: int
    integrator_idx: int | None

    _target_class: ClassVar[Type] = ResultSource

    @classmethod
    def _unstructure(cls, obj: ResultSource):
        return f"{obj.device_id},{obj.awg_id},{obj.integrator_idx}"

    @classmethod
    def _structure(cls, obj, _) -> ResultSource:
        device_id, awg_id, integrator_idx = obj.split(",")
        return ResultSource(
            device_id,
            int(awg_id),
            int(integrator_idx) if integrator_idx != "None" else None,
        )


@attrs.define
class ArtifactsCodegenModel:
    """Model for the `ArtifactsCodegen`."""

    # TODO: Consider implementing the rest of the `ArtifactsCodegen` fields in the model.
    result_handle_maps: dict[ResultSourceModel, list[set[str]]]
    initializations: list[InitializationModel]
    realtime_execution_init: list[RealtimeExecutionInitModel]
    oscillator_params: list[OscillatorParamModel]
    integrator_allocations: list[IntegratorAllocationModel]
    acquire_lengths: list[AcquireLengthModel]
    _target_class: ClassVar[Type] = ArtifactsCodegen


def _old_serialize(obj):
    return Serializer.to_dict(obj)


def _old_deserialize(obj, obj_type):
    return Serializer.load(obj, obj_type)


def make_converter():
    converter = make_laboneq_converter()
    register_models(converter, collect_models(sys.modules[__name__]))
    return converter


_converter = make_converter()

# List of fields that are modeled in `ArtifactsCodegenModel`. Other fields of `ArtifactsCodegen` are handled by the legacy serializer.
_MODELED_FIELDS = [field.name for field in ArtifactsCodegenModel.__attrs_attrs__]


def serialize_artifacts_qccs(artifacts: ArtifactsCodegen) -> dict[str, Any]:
    """Serialize `ArtifactsCodegen` to a dictionary."""
    legacy_artifacts = replace(artifacts, **dict.fromkeys(_MODELED_FIELDS, []))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        legacy = _old_serialize(legacy_artifacts)
    return {
        "_legacy": legacy,
        **_converter.unstructure(artifacts, ArtifactsCodegenModel),
    }


def deserialize_artifacts_qccs(data: dict[str, Any]) -> ArtifactsCodegen:
    """Deserialize a dictionary to `ArtifactsCodegen`."""
    data = dict(data)
    legacy_data = data.pop("_legacy")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        artifacts = _old_deserialize(legacy_data, ArtifactsCodegen)
    modeled = _converter.structure(data, ArtifactsCodegenModel)
    for key in _MODELED_FIELDS:
        setattr(artifacts, key, getattr(modeled, key))
    return artifacts


# Register ArtifactsCodegen as a CompilerArtifact plugin
register_compiler_artifact_plugin(
    "QCCS",
    ArtifactsCodegen,
    serialize_artifacts_qccs,
    deserialize_artifacts_qccs,
)
