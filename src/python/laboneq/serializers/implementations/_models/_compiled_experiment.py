# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Models for the compiled experiment."""

from __future__ import annotations

import sys
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Type, Union

import attrs

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
    SignalType,
    SoftwareVersions,
    TriggeringMode,
)
from laboneq.data.scheduled_experiment import (
    ArtifactsCodegen,
    CompilerArtifact,
    ScheduledExperiment,
)
from laboneq.dsl.serialization import Serializer
from laboneq.executor.executor import Statement

from ._common import (
    collect_models,
    make_laboneq_converter,
    register_models,
)

# Models for Recipe:


class SignalTypeModel(Enum):
    IQ = "iq"
    SINGLE = "single"
    INTEGRATION = "integration"
    MARKER = "marker"
    _target_class = SignalType


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
    signal_type: SignalTypeModel
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
    device_id: str | None
    awg_id: int | str
    program_ref: str
    nt_step: NtStepKeyModel
    wave_indices_ref: str | None
    kernel_indices_ref: str | None
    _target_class: ClassVar[Type] = RealtimeExecutionInit


@attrs.define
class SoftwareVersionsModel:
    target_labone: str
    laboneq: str
    _target_class: ClassVar[Type] = SoftwareVersions


@attrs.define
class RecipeModel:
    initializations: list[InitializationModel]
    realtime_execution_init: list[RealtimeExecutionInitModel]
    oscillator_params: list[OscillatorParamModel]
    integrator_allocations: list[IntegratorAllocationModel]
    acquire_lengths: list[AcquireLengthModel]
    simultaneous_acquires: list[dict[str, str]]
    total_execution_time: float
    max_step_execution_time: float
    is_spectroscopy: bool
    versions: SoftwareVersionsModel
    _target_class: ClassVar[Type] = Recipe


@attrs.define
class ScheduledExperimentModel:
    uid: str | None = None
    device_setup_fingerprint: str | None = None
    recipe: RecipeModel | None = None
    compilation_job_hash: str | None = None
    experiment_hash: str | None = None

    # NOTE! The data structure for the following fields is not completely
    # defined in the original code.
    # Resort to using the old serializer for now.
    # TODO: Revisit this later to swap out the old serializer
    # with the new one completely.
    artifacts: CompilerArtifact | None = None
    schedule: dict[str, Any] | None = None
    execution: Statement | None = None

    _target_class: ClassVar[Type] = ScheduledExperiment

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "device_setup_fingerprint": obj.device_setup_fingerprint,
            "recipe": _converter.unstructure(obj.recipe, RecipeModel),
            "compilation_job_hash": obj.compilation_job_hash,
            "experiment_hash": obj.experiment_hash,
            "artifacts": Serializer.to_dict(obj.artifacts),
            "schedule": Serializer.to_dict(obj.schedule),
            "execution": Serializer.to_dict(obj.execution),
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            device_setup_fingerprint=obj["device_setup_fingerprint"],
            recipe=_converter.structure(obj["recipe"], RecipeModel),
            compilation_job_hash=obj["compilation_job_hash"],
            experiment_hash=obj["experiment_hash"],
            artifacts=Serializer.load(obj["artifacts"], ArtifactsCodegen),
            schedule=Serializer.load(obj["schedule"], dict),
            execution=Serializer.load(obj["execution"], Statement),
        )


def make_converter():
    _converter = make_laboneq_converter()
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
