# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from enum import Enum
from functools import partial
from typing import ClassVar, Type, Union

import attrs
from cattrs import Converter

from laboneq.core.types.enums.carrier_type import CarrierType
from laboneq.core.types.enums.high_pass_compensation_clearing import (
    HighPassCompensationClearing,
)
from laboneq.core.types.enums.modulation_type import ModulationType
from laboneq.core.types.enums.port_mode import PortMode
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
from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.parameter import LinearSweepParameter, SweepParameter

from ._common import (
    ArrayLike_Model,
    collect_models,
    make_laboneq_converter,
    register_models,
    structure_union_generic_type,
    unstructure_union_generic_type,
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


#
# Attrs models for the serialized data:
#


@attrs.define
class SweepParameterModel:
    uid: str
    values: ArrayLike_Model | None
    axis_name: str | None
    driven_by: list[SweepParameterModel | LinearSweepParameterModel] | None
    _target_class: ClassVar[Type] = SweepParameter

    @classmethod
    def _unstructure(cls, obj):
        uid = obj.uid
        values = _converter.unstructure(obj.values, ArrayLike_Model)
        axis_name = obj.axis_name
        if obj.driven_by:
            driven_by = [
                _converter.unstructure(i, ParameterModel) for i in obj.driven_by
            ]
        else:
            driven_by = None
        return {
            "uid": uid,
            "values": values,
            "axis_name": axis_name,
            "driven_by": driven_by,
        }

    @classmethod
    def _structure(cls, d, _):
        if d["driven_by"] is None:
            driven_by = None
        else:
            driven_by = []
            for db in d["driven_by"]:
                if db["_type"] == "SweepParameterModel":
                    driven_by.append(_converter.structure(db, SweepParameterModel))
                else:
                    driven_by.append(_converter.structure(db, SweepParameterModel))
        return cls._target_class(
            uid=d["uid"],
            values=d["values"],
            axis_name=d["axis_name"],
            driven_by=driven_by,
        )


@attrs.define
class LinearSweepParameterModel:
    uid: str
    start: float | None
    stop: float | None
    count: int | None
    axis_name: str | None
    _target_class: ClassVar[Type] = LinearSweepParameter

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "start": obj.start,
            "stop": obj.stop,
            "count": obj.count,
            "axis_name": obj.axis_name,
        }

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            uid=d["uid"],
            start=d["start"],
            stop=d["stop"],
            count=d["count"],
            axis_name=d["axis_name"],
        )


ParameterModel = Union[SweepParameterModel, LinearSweepParameterModel]


def _unstructure_parameter_model(obj, _converter: Converter):
    return unstructure_union_generic_type(
        obj, [SweepParameterModel, LinearSweepParameterModel], _converter
    )


def _structure_parameter_model(d, _, _converter: Converter):
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(
        d, [SweepParameterModel, LinearSweepParameterModel], _converter
    )


def _unstructure_basic_or_parameter_model(obj, _converter: Converter):
    return (
        obj
        if obj is None or isinstance(obj, (float, int, complex))
        else _converter.unstructure(obj, ParameterModel)
    )


def _structure_basic_or_parameter_model(v, _converter: Converter):
    return (
        v
        if v is None or isinstance(v, (float, int, complex))
        else _converter.structure(v, ParameterModel)
    )


@attrs.define
class OscillatorModel:
    uid: str
    frequency: float | ParameterModel | None
    modulation_type: ModulationTypeModel
    carrier_type: CarrierTypeModel | None
    _target_class: ClassVar[Type] = Oscillator

    @classmethod
    def _unstructure(cls, obj):
        uid = obj.uid
        frequency = _unstructure_basic_or_parameter_model(obj.frequency, _converter)
        modulation_type = _converter.unstructure(
            obj.modulation_type, ModulationTypeModel
        )
        carrier_type = _converter.unstructure(
            obj.carrier_type, Union[CarrierTypeModel, None]
        )
        return {
            "uid": uid,
            "frequency": frequency,
            "modulation_type": modulation_type,
            "carrier_type": carrier_type,
        }

    @classmethod
    def _structure(cls, d, _):
        mod_type = ModulationTypeModel._target_class.value(d["modulation_type"])
        if d["carrier_type"] is None:
            carrier_type = None
        else:
            carrier_type = CarrierTypeModel._target_class.value(d["carrier_type"])
        return cls._target_class(
            uid=d["uid"],
            frequency=_structure_basic_or_parameter_model(d["frequency"], _converter),
            modulation_type=mod_type,
            carrier_type=carrier_type,
        )


@attrs.define
class MixerCalibrationModel:
    uid: str
    voltage_offsets: list[float | ParameterModel] | None
    correction_matrix: list[list[float | ParameterModel]] | None
    _target_class: ClassVar[Type] = MixerCalibration

    @classmethod
    def _unstructure(cls, obj):
        uid = obj.uid
        if obj.voltage_offsets:
            voltage_offsets = [
                _unstructure_basic_or_parameter_model(i, _converter)
                for i in obj.voltage_offsets
            ]
        else:
            voltage_offsets = None
        if obj.correction_matrix is None:
            correction_matrix = None
        else:
            correction_matrix = [
                [
                    i
                    if isinstance(i, (float, int))
                    else _converter.unstructure(i, ParameterModel)
                    for i in row
                ]
                for row in obj.correction_matrix
            ]
        return {
            "uid": uid,
            "voltage_offsets": voltage_offsets,
            "correction_matrix": correction_matrix,
        }

    @classmethod
    def _structure(cls, d, _):
        uid = d["uid"]
        if d["voltage_offsets"]:
            voltage_offsets = [
                _structure_basic_or_parameter_model(i, _converter)
                for i in d["voltage_offsets"]
            ]
        else:
            voltage_offsets = None
        if d["correction_matrix"]:
            correction_matrix = [
                [_structure_basic_or_parameter_model(i, _converter) for i in row]
                for row in d["correction_matrix"]
            ]
        else:
            correction_matrix = None
        return cls._target_class(
            uid=uid,
            voltage_offsets=voltage_offsets,
            correction_matrix=correction_matrix,
        )


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

    @classmethod
    def _unstructure(cls, obj):
        uid = obj.uid
        return {
            "uid": uid,
            "pump_frequency": _unstructure_basic_or_parameter_model(
                obj.pump_frequency, _converter
            ),
            "pump_power": _unstructure_basic_or_parameter_model(
                obj.pump_power, _converter
            ),
            "pump_on": obj.pump_on,
            "pump_filter_on": obj.pump_filter_on,
            "cancellation_on": obj.cancellation_on,
            "cancellation_phase": _unstructure_basic_or_parameter_model(
                obj.cancellation_phase, _converter
            ),
            "cancellation_attenuation": _unstructure_basic_or_parameter_model(
                obj.cancellation_attenuation, _converter
            ),
            "cancellation_source": _converter.unstructure(
                obj.cancellation_source, CancellationSourceModel
            ),
            "cancellation_source_frequency": obj.cancellation_source_frequency,
            "alc_on": obj.alc_on,
            "probe_on": obj.probe_on,
            "probe_frequency": _unstructure_basic_or_parameter_model(
                obj.probe_frequency, _converter
            ),
            "probe_power": _unstructure_basic_or_parameter_model(
                obj.probe_power, _converter
            ),
        }

    @classmethod
    def _structure(cls, d, _):
        pump_frequency = _structure_basic_or_parameter_model(
            d["pump_frequency"], _converter
        )
        pump_power = _structure_basic_or_parameter_model(d["pump_power"], _converter)
        cancellation_phase = _structure_basic_or_parameter_model(
            d["cancellation_phase"], _converter
        )
        cancellation_attenuation = _structure_basic_or_parameter_model(
            d["cancellation_attenuation"], _converter
        )
        probe_frequency = _structure_basic_or_parameter_model(
            d["probe_frequency"], _converter
        )
        probe_power = _structure_basic_or_parameter_model(d["probe_power"], _converter)
        return cls._target_class(
            uid=d["uid"],
            pump_frequency=pump_frequency,
            pump_power=pump_power,
            pump_on=d["pump_on"],
            pump_filter_on=d["pump_filter_on"],
            cancellation_on=d["cancellation_on"],
            cancellation_phase=cancellation_phase,
            cancellation_attenuation=cancellation_attenuation,
            cancellation_source=CancellationSourceModel._target_class.value(
                d["cancellation_source"]
            ),
            cancellation_source_frequency=d["cancellation_source_frequency"],
            alc_on=d["alc_on"],
            probe_on=d["probe_on"],
            probe_frequency=probe_frequency,
            probe_power=probe_power,
        )


@attrs.define
class OutputRouteModel:
    source: str
    amplitude_scaling: float | ParameterModel
    phase_shift: float | ParameterModel | None
    _target_class: ClassVar[Type] = OutputRoute
    # TODO: source is in fact str | LogicalSignalModel | ExperimentSignalModel
    # Including the above two models would create a circular dependency
    # between the models. Use str for now.

    @classmethod
    def _unstructure(cls, obj):
        if isinstance(obj.source, LogicalSignal):
            source = obj.source.path
        elif isinstance(obj.source, ExperimentSignal):
            source = obj.source.mapped_logical_signal_path
        else:
            source = obj.source
        return {
            "source": source,
            "amplitude_scaling": _unstructure_basic_or_parameter_model(
                obj.amplitude_scaling, _converter
            ),
            "phase_shift": _unstructure_basic_or_parameter_model(
                obj.phase_shift, _converter
            ),
        }

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            source=d["source"],
            amplitude_scaling=_structure_basic_or_parameter_model(
                d["amplitude_scaling"], _converter
            ),
            phase_shift=_structure_basic_or_parameter_model(
                d["phase_shift"], _converter
            ),
        )


@attrs.define
class PrecompensationModel:
    uid: str
    exponential: list[ExponentialCompensationModel] | None
    high_pass: HighPassCompensationModel | None
    bounce: BounceCompensationModel | None
    FIR: FIRCompensationModel | None
    _target_class: ClassVar[Type] = Precompensation

    @classmethod
    def _unstructure(cls, obj):
        if obj.exponential:
            exponential = [
                _converter.unstructure(i, ExponentialCompensationModel)
                for i in obj.exponential
            ]
        else:
            exponential = None
        high_pass = (
            _converter.unstructure(obj.high_pass, HighPassCompensationModel)
            if obj.high_pass
            else None
        )
        bounce = (
            _converter.unstructure(obj.bounce, BounceCompensationModel)
            if obj.bounce
            else None
        )
        fir = _converter.unstructure(obj.FIR, FIRCompensationModel) if obj.FIR else None
        return {
            "uid": obj.uid,
            "exponential": exponential,
            "high_pass": high_pass,
            "bounce": bounce,
            "FIR": fir,
        }

    @classmethod
    def _structure(cls, d, _):
        exponential = (
            None
            if d["exponential"] is None
            else [
                _converter.structure(i, ExponentialCompensationModel)
                for i in d["exponential"]
            ]
        )
        high_pass = (
            _converter.structure(d["high_pass"], HighPassCompensationModel)
            if d["high_pass"]
            else None
        )
        bounce = (
            _converter.structure(d["bounce"], BounceCompensationModel)
            if d["bounce"]
            else None
        )
        fir = _converter.structure(d["FIR"], FIRCompensationModel) if d["FIR"] else None
        return cls._target_class(
            uid=d["uid"],
            exponential=exponential,
            high_pass=high_pass,
            bounce=bounce,
            FIR=fir,
        )


@attrs.define
class FIRCompensationModel:
    coefficients: ArrayLike_Model
    _target_class: ClassVar[Type] = FIRCompensation

    @classmethod
    def _unstructure(cls, obj):
        return {
            "coefficients": _converter.unstructure(obj.coefficients, ArrayLike_Model)
        }

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            coefficients=_converter.structure(d["coefficients"], ArrayLike_Model)
        )


@attrs.define
class ExponentialCompensationModel:
    timeconstant: float
    amplitude: float
    _target_class: ClassVar[Type] = ExponentialCompensation

    @classmethod
    def _unstructure(cls, obj):
        return {"timeconstant": obj.timeconstant, "amplitude": obj.amplitude}

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            timeconstant=d["timeconstant"], amplitude=d["amplitude"]
        )


@attrs.define
class HighPassCompensationModel:
    timeconstant: float
    clearing: HighPassCompensationClearingModel | None
    _target_class: ClassVar[Type] = HighPassCompensation

    @classmethod
    def _unstructure(cls, obj):
        clearing = (
            _converter.unstructure(obj.clearing, HighPassCompensationClearingModel)
            if obj.clearing
            else None
        )
        return {"timeconstant": obj.timeconstant, "clearing": clearing}

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            timeconstant=d["timeconstant"],
            clearing=HighPassCompensationClearingModel._target_class.value(
                d["clearing"]
            )
            if d["clearing"]
            else None,
        )


@attrs.define
class BounceCompensationModel:
    delay: float
    amplitude: float
    _target_class: ClassVar[Type] = BounceCompensation

    @classmethod
    def _unstructure(cls, obj):
        return {"delay": obj.delay, "amplitude": obj.amplitude}

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(delay=d["delay"], amplitude=d["amplitude"])


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
    range: int | float | None
    threshold: float | list[float] | None
    amplitude: float | ParameterModel | None
    amplifier_pump: AmplifierPumpModel | None
    added_outputs: list[OutputRouteModel] | None
    automute: bool
    _target_class: ClassVar[Type] = SignalCalibration

    @classmethod
    def _unstructure(cls, obj):
        osc = (
            _converter.unstructure(obj.oscillator, OscillatorModel)
            if obj.oscillator
            else None
        )
        local_osc = (
            _converter.unstructure(obj.local_oscillator, OscillatorModel)
            if obj.local_oscillator
            else None
        )
        mixer_calib = (
            _converter.unstructure(obj.mixer_calibration, MixerCalibrationModel)
            if obj.mixer_calibration
            else None
        )
        precomp = (
            _converter.unstructure(obj.precompensation, PrecompensationModel)
            if obj.precompensation
            else None
        )
        port_delay = (
            obj.port_delay
            if obj.port_delay is None or isinstance(obj.port_delay, (float, int))
            else _converter.unstructure(obj.port_delay, ParameterModel)
        )
        port_mode = (
            _converter.unstructure(obj.port_mode, PortModeModel)
            if obj.port_mode
            else None
        )
        delay_signal = obj.delay_signal
        voltage_offset = (
            obj.voltage_offset
            if obj.voltage_offset is None
            or isinstance(obj.voltage_offset, (float, int))
            else _converter.unstructure(obj.voltage_offset, ParameterModel)
        )
        range = obj.range
        threshold = obj.threshold
        amplitude = (
            obj.amplitude
            if obj.amplitude is None or isinstance(obj.amplitude, (float, int))
            else _converter.unstructure(obj.amplitude, ParameterModel)
        )
        amplifier_pump = (
            obj.amplifier_pump
            if obj.amplifier_pump is None
            else _converter.unstructure(obj.amplifier_pump, AmplifierPumpModel)
        )
        added_outputs = (
            None
            if obj.added_outputs is None
            else [
                _converter.unstructure(i, OutputRouteModel) for i in obj.added_outputs
            ]
        )
        automute = obj.automute
        return {
            "oscillator": osc,
            "local_oscillator": local_osc,
            "mixer_calibration": mixer_calib,
            "precompensation": precomp,
            "port_delay": port_delay,
            "port_mode": port_mode,
            "delay_signal": delay_signal,
            "voltage_offset": voltage_offset,
            "range": range,
            "threshold": threshold,
            "amplitude": amplitude,
            "amplifier_pump": amplifier_pump,
            "added_outputs": added_outputs,
            "automute": automute,
        }

    @classmethod
    def _structure(cls, d, _):
        osc = (
            _converter.structure(d["oscillator"], OscillatorModel)
            if d["oscillator"]
            else None
        )
        local_osc = (
            _converter.structure(d["local_oscillator"], OscillatorModel)
            if d["local_oscillator"]
            else None
        )
        mixer_calib = (
            _converter.structure(d["mixer_calibration"], MixerCalibrationModel)
            if d["mixer_calibration"]
            else None
        )
        precomp = (
            _converter.structure(d["precompensation"], PrecompensationModel)
            if d["precompensation"]
            else None
        )
        port_delay = (
            d["port_delay"]
            if d["port_delay"] is None or isinstance(d["port_delay"], (float, int))
            else _converter.structure(d["port_delay"], ParameterModel)
        )
        port_mode = (
            _converter.structure(d["port_mode"], PortModeModel)
            if d["port_mode"]
            else None
        )
        delay_signal = d["delay_signal"]
        voltage_offset = _structure_basic_or_parameter_model(
            d["voltage_offset"], _converter
        )
        range = d["range"]
        threshold = d["threshold"]
        amplitude = _structure_basic_or_parameter_model(d["amplitude"], _converter)
        amplifier_pump = (
            d["amplifier_pump"]
            if d["amplifier_pump"] is None
            else _converter.structure(d["amplifier_pump"], AmplifierPumpModel)
        )
        added_outputs = (
            None
            if d["added_outputs"] is None
            else [_converter.structure(i, OutputRouteModel) for i in d["added_outputs"]]
        )
        return cls._target_class(
            oscillator=osc,
            local_oscillator=local_osc,
            mixer_calibration=mixer_calib,
            precompensation=precomp,
            port_delay=port_delay,
            port_mode=port_mode,
            delay_signal=delay_signal,
            voltage_offset=voltage_offset,
            range=range,
            threshold=threshold,
            amplitude=amplitude,
            amplifier_pump=amplifier_pump,
            added_outputs=added_outputs,
            automute=d["automute"],
        )


@attrs.define
class CalibrationModel:
    calibration_items: dict[str, SignalCalibrationModel]
    _target_class: ClassVar[Type] = Calibration

    @classmethod
    def _unstructure(cls, obj):
        return {
            "calibration_items": {
                k: _converter.unstructure(v, SignalCalibrationModel)
                for k, v in obj.calibration_items.items()
            }
        }

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            calibration_items={
                k: _converter.structure(v, SignalCalibrationModel)
                for k, v in d["calibration_items"].items()
            }
        )


def make_converter():
    _converter = make_laboneq_converter()
    _converter.register_unstructure_hook(
        ParameterModel,
        partial(_unstructure_parameter_model, _converter=_converter),
    )
    _converter.register_structure_hook(
        ParameterModel,
        partial(_structure_parameter_model, _converter=_converter),
    )
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
