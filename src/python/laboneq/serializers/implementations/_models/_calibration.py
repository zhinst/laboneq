# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from enum import Enum
from typing import ClassVar, Type, Union

import attrs
import numpy
from cattrs import Converter

from laboneq.core.types.enums.carrier_type import CarrierType
from laboneq.core.types.enums.high_pass_compensation_clearing import (
    HighPassCompensationClearing,
)
from laboneq.core.types.enums.modulation_type import ModulationType
from laboneq.core.types.enums.port_mode import PortMode
from laboneq.core.types.units import Quantity, Unit
from laboneq.dsl.calibration import CancellationSource
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.calibration import Calibration
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.output_routing import OutputRoute
from laboneq.dsl.calibration.precompensation import (
    BounceCompensation,
    ExponentialCompensation,
    FIRCompensation,
    HighPassCompensation,
    Precompensation,
)
from laboneq.dsl.calibration.signal_calibration import SignalCalibration
from laboneq.dsl.parameter import LinearSweepParameter, SweepParameter

from ._common import (
    ArrayLike_Model,
    collect_models,
    make_laboneq_converter,
    register_models,
)


class CancellationSourceModel(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    _target_class = CancellationSource


class CarrierTypeModel(Enum):
    IF = "INTERMEDIATE_FREQUENCY"
    RF = "RADIO_FREQUENCY"
    _target_class = CarrierType


class ModulationTypeModel(Enum):
    SOFTWARE = "SOFTWARE"
    HARDWARE = "HARDWARE"
    AUTO = "AUTO"
    _target_class = ModulationType


class PortModeModel(Enum):
    LF = "LF"
    RF = "RF"
    _target_class = PortMode


class HighPassCompensationClearingModel(Enum):
    LEVEL = "LEVEL"
    RISE = "RISE"
    FALL = "FALL"
    BOTH = "BOTH"
    _target_class = HighPassCompensationClearing


class UnitModel(Enum):
    volt = "volt"
    dBm = "dBm"
    _target_class = Unit


#
# Attrs models for the serialized data:
#


@attrs.define
class SweepParameterModel:
    uid: str
    values: ArrayLike_Model | None
    axis_name: str | None
    driven_by: list[ParameterModel] | None
    _target_class: ClassVar[Type] = SweepParameter


@attrs.define
class LinearSweepParameterModel:
    uid: str
    start: float | None
    stop: float | None
    count: int | None
    axis_name: str | None
    _target_class: ClassVar[Type] = LinearSweepParameter


ParameterModel = Union[SweepParameterModel, LinearSweepParameterModel]


def unstructure_basic_or_parameter_model(obj, _converter: Converter):
    if obj is None:
        return None
    if isinstance(obj, (float, int, complex, numpy.number)):
        return _converter.unstructure(obj, float | int | complex | numpy.number)
    else:
        return _converter.unstructure(obj, ParameterModel)


def structure_basic_or_parameter_model(v, _converter: Converter):
    if v is None:
        return None
    elif isinstance(v, (float, int, numpy.number)) or isinstance(v, list):
        # we unstructure complex and numpy complex to a list of two numbers.
        return _converter.structure(v, float | int | complex | numpy.number)
    else:
        return _converter.structure(v, ParameterModel)


@attrs.define
class OscillatorModel:
    uid: str
    frequency: float | ParameterModel | None
    modulation_type: ModulationTypeModel
    carrier_type: CarrierTypeModel | None
    _target_class: ClassVar[Type] = Oscillator


@attrs.define
class MixerCalibrationModel:
    uid: str
    voltage_offsets: list[float | ParameterModel] | None
    correction_matrix: list[list[float | ParameterModel]] | None
    _target_class: ClassVar[Type] = MixerCalibration


@attrs.define
class AmplifierPumpModel:
    uid: str
    pump_frequency: float | ParameterModel | None
    pump_power: float | ParameterModel | None
    pump_on: bool
    pump_filter_on: bool
    cancellation_on: bool
    cancellation_phase: float | ParameterModel | None
    cancellation_attenuation: float | ParameterModel | None
    cancellation_source: CancellationSourceModel
    cancellation_source_frequency: float | None
    alc_on: bool
    probe_on: bool
    probe_frequency: float | ParameterModel | None
    probe_power: float | ParameterModel | None
    _target_class: ClassVar[Type] = AmplifierPump


@attrs.define
class OutputRouteModel:
    source: str
    amplitude_scaling: float | ParameterModel
    phase_shift: float | ParameterModel | None
    _target_class: ClassVar[Type] = OutputRoute
    # TODO: source is in fact str | LogicalSignalModel | ExperimentSignalModel
    # Including the above two models would create a circular dependency
    # between the models. Use str for now.


@attrs.define
class PrecompensationModel:
    uid: str
    exponential: list[ExponentialCompensationModel] | None
    high_pass: HighPassCompensationModel | None
    bounce: BounceCompensationModel | None
    FIR: FIRCompensationModel | None
    _target_class: ClassVar[Type] = Precompensation


@attrs.define
class FIRCompensationModel:
    coefficients: ArrayLike_Model
    _target_class: ClassVar[Type] = FIRCompensation


@attrs.define
class ExponentialCompensationModel:
    timeconstant: float
    amplitude: float
    _target_class: ClassVar[Type] = ExponentialCompensation


@attrs.define
class HighPassCompensationModel:
    timeconstant: float
    clearing: HighPassCompensationClearingModel | None
    _target_class: ClassVar[Type] = HighPassCompensation


@attrs.define
class BounceCompensationModel:
    delay: float
    amplitude: float
    _target_class: ClassVar[Type] = BounceCompensation


@attrs.define
class QuantityModel:
    value: float
    unit: UnitModel
    _target_class: ClassVar[Type] = Quantity


@attrs.define
class SignalCalibrationModel:
    oscillator: OscillatorModel | None
    local_oscillator: OscillatorModel | None
    mixer_calibration: MixerCalibrationModel | None
    precompensation: PrecompensationModel | None
    port_delay: float | ParameterModel | None
    port_mode: PortModeModel | None
    delay_signal: float | None
    voltage_offset: float | ParameterModel | None
    range: int | float | QuantityModel | None
    threshold: float | list[float] | None
    amplitude: float | ParameterModel | None
    amplifier_pump: AmplifierPumpModel | None
    added_outputs: list[OutputRouteModel] | None
    automute: bool
    _target_class: ClassVar[Type] = SignalCalibration


@attrs.define
class CalibrationModel:
    calibration_items: dict[str, SignalCalibrationModel]
    _target_class: ClassVar[Type] = Calibration


def make_converter():
    _converter = make_laboneq_converter()
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter
