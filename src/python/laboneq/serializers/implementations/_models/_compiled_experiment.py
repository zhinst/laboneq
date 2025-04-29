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

    # simple enough to not require a customized unstructuring method

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(indices=obj["indices"])


@attrs.define
class GainsModel:
    diagonal: Union[float, str]
    off_diagonal: Union[float, str]
    _target_class: ClassVar[Type] = Gains

    # simple enough to not require a customized unstructuring method
    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            diagonal=obj["diagonal"], off_diagonal=obj["off_diagonal"]
        )


@attrs.define
class RoutedOutputModel:
    from_channel: int
    amplitude: Union[float, str]
    phase: Union[float, str]
    _target_class: ClassVar[Type] = RoutedOutput

    # simple enough to not require a customized unstructuring method
    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            from_channel=obj["from_channel"],
            amplitude=obj["amplitude"],
            phase=obj["phase"],
        )


@attrs.define
class IOModel:
    channel: int
    enable: bool | None
    modulation: bool | None
    offset: float | str | None
    gains: GainsModel | None
    range: float | None
    range_unit: str | None
    precompensation: dict[str, dict] | None
    lo_frequency: float | None
    port_mode: str | None
    port_delay: float | None
    scheduler_port_delay: float
    delay_signal: float | None
    marker_mode: str | None
    amplitude: float | None
    routed_outputs: list[RoutedOutputModel]
    enable_output_mute: bool
    _target_class: ClassVar[Type] = IO

    @classmethod
    def _unstructure(cls, obj):
        return {
            "channel": obj.channel,
            "enable": obj.enable,
            "modulation": obj.modulation,
            "offset": obj.offset,
            "gains": _converter.unstructure(obj.gains, Optional[GainsModel]),
            "range": obj.range,
            "range_unit": obj.range_unit,
            "precompensation": obj.precompensation,
            "lo_frequency": obj.lo_frequency,
            "port_mode": obj.port_mode,
            "port_delay": obj.port_delay,
            "scheduler_port_delay": obj.scheduler_port_delay,
            "delay_signal": obj.delay_signal,
            "marker_mode": obj.marker_mode,
            "amplitude": obj.amplitude,
            "routed_outputs": [
                _converter.unstructure(ro, RoutedOutputModel)
                for ro in obj.routed_outputs
            ],
            "enable_output_mute": obj.enable_output_mute,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            channel=obj["channel"],
            enable=obj["enable"],
            modulation=obj["modulation"],
            offset=obj["offset"],
            gains=_converter.structure(obj["gains"], Optional[GainsModel]),
            range=obj["range"],
            range_unit=obj["range_unit"],
            precompensation=obj["precompensation"],
            lo_frequency=obj["lo_frequency"],
            port_mode=obj["port_mode"],
            port_delay=obj["port_delay"],
            scheduler_port_delay=obj["scheduler_port_delay"],
            delay_signal=obj["delay_signal"],
            marker_mode=obj["marker_mode"],
            amplitude=obj["amplitude"],
            routed_outputs=[
                _converter.structure(ro, RoutedOutputModel)
                for ro in obj["routed_outputs"]
            ],
            enable_output_mute=obj["enable_output_mute"],
        )


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

    @classmethod
    def _unstructure(cls, obj):
        return {
            "awg": obj.awg,
            "signal_type": _converter.unstructure(obj.signal_type, SignalTypeModel),
            "signals": obj.signals,
            "source_feedback_register": obj.source_feedback_register,
            "codeword_bitshift": obj.codeword_bitshift,
            "codeword_bitmask": obj.codeword_bitmask,
            "feedback_register_index_select": obj.feedback_register_index_select,
            "command_table_match_offset": obj.command_table_match_offset,
            "target_feedback_register": obj.target_feedback_register,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            awg=obj["awg"],
            signal_type=SignalTypeModel._target_class.value(obj["signal_type"]),
            signals=obj["signals"],
            source_feedback_register=obj["source_feedback_register"],
            codeword_bitshift=obj["codeword_bitshift"],
            codeword_bitmask=obj["codeword_bitmask"],
            feedback_register_index_select=obj["feedback_register_index_select"],
            command_table_match_offset=obj["command_table_match_offset"],
            target_feedback_register=obj["target_feedback_register"],
        )


@attrs.define
class MeasurementModel:
    length: int
    channel: int = 0
    _target_class: ClassVar[Type] = Measurement

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(length=obj["length"], channel=obj["channel"])


@attrs.define
class ConfigModel:
    repetitions: int
    triggering_mode: TriggeringModeModel
    sampling_rate: float | None
    _target_class: ClassVar[Type] = Config

    @classmethod
    def _unstructure(cls, obj):
        return {
            "repetitions": obj.repetitions,
            "triggering_mode": _converter.unstructure(
                obj.triggering_mode, TriggeringModeModel
            ),
            "sampling_rate": obj.sampling_rate,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            repetitions=obj["repetitions"],
            triggering_mode=TriggeringModeModel._target_class.value(
                obj["triggering_mode"]
            ),
            sampling_rate=obj["sampling_rate"],
        )


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

    @classmethod
    def _unstructure(cls, obj):
        return {
            "device_uid": obj.device_uid,
            "device_type": obj.device_type,
            "config": _converter.unstructure(obj.config, ConfigModel),
            "awgs": [_converter.unstructure(awg, AWGModel) for awg in obj.awgs],
            "outputs": [
                _converter.unstructure(output, IOModel) for output in obj.outputs
            ],
            "inputs": [_converter.unstructure(input, IOModel) for input in obj.inputs],
            "measurements": [
                _converter.unstructure(m, MeasurementModel) for m in obj.measurements
            ],
            "ppchannels": obj.ppchannels,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            device_uid=obj["device_uid"],
            device_type=obj["device_type"],
            config=_converter.structure(obj["config"], ConfigModel),
            awgs=[_converter.structure(awg, AWGModel) for awg in obj["awgs"]],
            outputs=[
                _converter.structure(output, IOModel) for output in obj["outputs"]
            ],
            inputs=[_converter.structure(input, IOModel) for input in obj["inputs"]],
            measurements=[
                _converter.structure(m, MeasurementModel) for m in obj["measurements"]
            ],
            ppchannels=obj["ppchannels"],
        )


@attrs.define
class OscillatorParamModel:
    id: str
    device_id: str
    channel: int
    signal_id: str
    frequency: Optional[float]
    param: Optional[str]
    _target_class: ClassVar[Type] = OscillatorParam

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            id=obj["id"],
            device_id=obj["device_id"],
            channel=obj["channel"],
            signal_id=obj["signal_id"],
            frequency=obj["frequency"],
            param=obj["param"],
        )


@attrs.define
class IntegratorAllocationModel:
    signal_id: str
    device_id: str
    awg: int
    channels: list[int]
    kernel_count: int
    thresholds: list[float]
    _target_class: ClassVar[Type] = IntegratorAllocation

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            signal_id=obj["signal_id"],
            device_id=obj["device_id"],
            awg=obj["awg"],
            channels=obj["channels"],
            kernel_count=obj["kernel_count"],
            thresholds=obj["thresholds"],
        )


@attrs.define
class AcquireLengthModel:
    signal_id: str
    acquire_length: int
    _target_class: ClassVar[Type] = AcquireLength

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            signal_id=obj["signal_id"],
            acquire_length=obj["acquire_length"],
        )


@attrs.define
class RealtimeExecutionInitModel:
    device_id: str | None
    awg_id: int | str
    program_ref: str
    nt_step: NtStepKeyModel
    wave_indices_ref: str | None
    kernel_indices_ref: str | None
    _target_class: ClassVar[Type] = RealtimeExecutionInit

    @classmethod
    def _unstructure(cls, obj):
        return {
            "device_id": obj.device_id,
            "awg_id": obj.awg_id,
            "program_ref": obj.program_ref,
            "nt_step": _converter.unstructure(obj.nt_step, NtStepKeyModel),
            "wave_indices_ref": obj.wave_indices_ref,
            "kernel_indices_ref": obj.kernel_indices_ref,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            device_id=obj["device_id"],
            awg_id=obj["awg_id"],
            program_ref=obj["program_ref"],
            nt_step=_converter.structure(obj["nt_step"], NtStepKeyModel),
            wave_indices_ref=obj["wave_indices_ref"],
            kernel_indices_ref=obj["kernel_indices_ref"],
        )


@attrs.define
class SoftwareVersionsModel:
    target_labone: str
    laboneq: str
    _target_class: ClassVar[Type] = SoftwareVersions

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            target_labone=obj["target_labone"],
            laboneq=obj["laboneq"],
        )


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

    @classmethod
    def _unstructure(cls, obj):
        return {
            "initializations": [
                _converter.unstructure(init, InitializationModel)
                for init in obj.initializations
            ],
            "realtime_execution_init": [
                _converter.unstructure(init, RealtimeExecutionInitModel)
                for init in obj.realtime_execution_init
            ],
            "oscillator_params": [
                _converter.unstructure(param, OscillatorParamModel)
                for param in obj.oscillator_params
            ],
            "integrator_allocations": [
                _converter.unstructure(alloc, IntegratorAllocationModel)
                for alloc in obj.integrator_allocations
            ],
            "acquire_lengths": [
                _converter.unstructure(length, AcquireLengthModel)
                for length in obj.acquire_lengths
            ],
            "simultaneous_acquires": obj.simultaneous_acquires,
            "total_execution_time": obj.total_execution_time,
            "max_step_execution_time": obj.max_step_execution_time,
            "is_spectroscopy": obj.is_spectroscopy,
            "versions": _converter.unstructure(obj.versions, SoftwareVersionsModel),
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            initializations=[
                _converter.structure(init, InitializationModel)
                for init in obj["initializations"]
            ],
            realtime_execution_init=[
                _converter.structure(init, RealtimeExecutionInitModel)
                for init in obj["realtime_execution_init"]
            ],
            oscillator_params=[
                _converter.structure(param, OscillatorParamModel)
                for param in obj["oscillator_params"]
            ],
            integrator_allocations=[
                _converter.structure(alloc, IntegratorAllocationModel)
                for alloc in obj["integrator_allocations"]
            ],
            acquire_lengths=[
                _converter.structure(length, AcquireLengthModel)
                for length in obj["acquire_lengths"]
            ],
            simultaneous_acquires=obj["simultaneous_acquires"],
            total_execution_time=obj["total_execution_time"],
            max_step_execution_time=obj["max_step_execution_time"],
            is_spectroscopy=obj["is_spectroscopy"],
            versions=_converter.structure(obj["versions"], SoftwareVersionsModel),
        )


@attrs.define
class ScheduledExperimentModel:
    uid: str | None = None
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
