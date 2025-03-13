# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


"""
LabOneQ data classes could be divided into two categories:
1. Data classes that are used as top-level objects, that can be serialized directly by users.
2. Data classes that are used as attributes of other data classes.

Only top-level data classes are versioned and have support for backward compatibility in serialization.
When one or several of the attributes of a top-level data class are of a Tier-2 data class, the serialization
shall be able to handle the recursive serialization of the nested data classes and bump the version
when the nested data classes are updated.

To solve this problem, we introduce the concept of "models" in the serialization process.
The models are used as a up-to-date "blueprint" for LabOneQ data classes.
When the data classes are updated, the discrepancy between the data classes and the models would
cause the serialization of some of the top-level data classes to fail, prompting the developers to update
the models and bump the version of the serializers.

Models shall be either attrs classes or enums and shall have an attribute `_target_class` that
points to the corresponding class that the model represents. The purpose of the `_target_class` attribute
are two-fold:
1. It indicates the class that the model represents, rather than relying on the class name.
2. It provides a way to convert the model to the corresponding class during deserialization.

**Notes on the implementation:**

The original ideas is to use the `cattrs` library to serialize the data using the models in this module,
then deserialize the dict using the models again (not the original classes because cattrs requires
all the type classes available at the time of deserialization, which is not the case for some of the
LabOneQ modules as we hide the imports into TYPECHECKING clauses to avoid the circular import).
The models are then converted to the original classes.
In this approach, the se/deserialization is automatic and would use:
make_dict_(un)structure_fn and register_(un)structure_hook in cattrs, which requires internal type evaluation.
Another example is Union [attrs, attrs] which fetch the hook for each type and hence also does type evaluation.

This lean method unfortunately does not work with for all cases (especially for classes with attrs that
have complicated and nested types) in Python 3.9.
One example is that the union of 'types.GenericAlias' and 'types.GenericAlias' or None type is not supported
in python 3.9

The current implementation is a manual approach where we define the unstructure and structure methods and should work
on all Python versions >= 3.9.
The downside is that we have to manually define the unstructure and structure methods for each model.
Using predicates helps to avoid type evaluation but could reduce performance a bit.

When removing support for Python 3.9, we should switch to the automatic approach using cattrs by:
- Remove the unstructure and structure methods from the models.
- Register the models using cattrs.register_structure_hook and cattrs.register_unstructure_hook.
"""

from __future__ import annotations
from collections.abc import Iterable
import attrs
import sys
import inspect

from enum import Enum
from typing import ClassVar, Type, Union

from cattr import Converter
from laboneq.core.types.enums.carrier_type import CarrierType
from laboneq.core.types.enums.high_pass_compensation_clearing import (
    HighPassCompensationClearing,
)
from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.core.types.enums.modulation_type import ModulationType
from laboneq.core.types.enums.port_mode import PortMode
from laboneq.core.types.enums.reference_clock_source import ReferenceClockSource
from laboneq.dsl.calibration import CancellationSource
from laboneq.dsl.calibration.amplifier_pump import AmplifierPump
from laboneq.dsl.calibration.calibration import Calibration
from laboneq.dsl.calibration.mixer_calibration import MixerCalibration
from laboneq.dsl.calibration.oscillator import Oscillator
from laboneq.dsl.calibration.output_routing import OutputRoute
from laboneq.dsl.calibration.precompensation import (
    ExponentialCompensation,
    FIRCompensation,
    HighPassCompensation,
    Precompensation,
    BounceCompensation,
)
from laboneq.dsl.calibration.signal_calibration import SignalCalibration
from laboneq.dsl.device.connection import Connection
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument
from laboneq.dsl.device.instruments.hdawg import HDAWG
from laboneq.dsl.device.instruments.pqsc import PQSC
from laboneq.dsl.device.instruments.shfppc import SHFPPC
from laboneq.dsl.device.instruments.shfqa import SHFQA
from laboneq.dsl.device.instruments.shfqc import SHFQC
from laboneq.dsl.device.instruments.shfsg import SHFSG
from laboneq.dsl.device.instruments.uhfqa import UHFQA
from laboneq.dsl.device.io_units.logical_signal import LogicalSignal
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannel,
    PhysicalChannelType,
)
from laboneq.dsl.device.logical_signal_group import LogicalSignalGroup
from laboneq.dsl.device.physical_channel_group import PhysicalChannelGroup
from laboneq.dsl.device.servers.data_server import DataServer
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.parameter import LinearSweepParameter, SweepParameter
from cattrs.gen import make_dict_unstructure_fn, make_dict_structure_fn
from cattrs.preconf.orjson import make_converter
import numpy

#
# Enum models for the serialized data:
#


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


class IODirectionModel(Enum):
    IN = "IN"
    OUT = "OUT"
    _target_class = IODirection


class PhysicalChannelTypeModel(Enum):
    IQ_CHANNEL = "iq_channel"
    RF_CHANNEL = "rf_channel"
    _target_class = PhysicalChannelType


class IOSignalTypeModel(Enum):
    I = "I"
    Q = "Q"
    IQ = "IQ"
    RF = "RF"
    SINGLE = "SINGLE"
    LO = "LO"
    DIO = "DIO"
    ZSYNC = "ZSYNC"
    PPC = "PPC"
    _target_class = IOSignalType


class ReferenceClockSourceModel(Enum):
    INTERNAL = "internal"
    EXTERNAL = "external"
    _target_class = ReferenceClockSource


#
# Here goes the "funky" types
#

# cattrs does not work with typing.TypeAlias, which is the true identity of
# np.typing.ArrayLike. Ideally, np.typing.ArrayLike should be as follows:
# ArrayLike_Model = Union[
#     numpy._typing._nested_sequence._NestedSequence[
#         Union[bool, int, float, complex, str, bytes]
#     ],
#     numpy._typing._array_like._Buffer,
#     numpy._typing._array_like._SupportsArray[numpy.dtype[Any]],
#     numpy._typing._nested_sequence._NestedSequence[
#         numpy._typing._array_like._SupportsArray[numpy.dtype[Any]]
#     ],
#     complex,
#     bytes,
# ]
# But we will be more practical and use the following:
ArrayLike_Model = Union[
    numpy.ndarray,
    list[Union[bool, int, float, complex, str, bytes]],
]

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


def _unstructure_parameter_model(obj):
    if isinstance(obj, SweepParameter):
        d = _converter.unstructure(obj, SweepParameterModel)
    elif isinstance(obj, LinearSweepParameter):
        d = _converter.unstructure(obj, LinearSweepParameterModel)
    else:
        raise ValueError(
            f"Unsupported parameter type: {type(obj)} when unstructuring ParameterModel"
        )
    return {"_type": type(obj).__name__, **d}


def _structure_parameter_model(d, _):
    dtype = d.get("_type", None)
    if dtype is None:
        raise ValueError("ParameterModel obj is missing the _type field")
    elif dtype == "SweepParameter":
        return _converter.structure(d, SweepParameterModel)
    else:
        return _converter.structure(d, LinearSweepParameterModel)


def _unstructure_basic_or_parameter_model(obj):
    return (
        obj
        if obj is None or isinstance(obj, (float, int))
        else _unstructure_parameter_model(obj)
    )


def _structure_basic_or_parameter_model(v):
    return (
        v
        if v is None or isinstance(v, (float, int))
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
        frequency = _unstructure_basic_or_parameter_model(obj.frequency)
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
            frequency=_structure_basic_or_parameter_model(d["frequency"]),
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
                _unstructure_basic_or_parameter_model(i) for i in obj.voltage_offsets
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
                _structure_basic_or_parameter_model(i) for i in d["voltage_offsets"]
            ]
        else:
            voltage_offsets = None
        if d["correction_matrix"]:
            correction_matrix = [
                [_structure_basic_or_parameter_model(i) for i in row]
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
            "pump_frequency": _unstructure_basic_or_parameter_model(obj.pump_frequency),
            "pump_power": _unstructure_basic_or_parameter_model(obj.pump_power),
            "pump_on": obj.pump_on,
            "pump_filter_on": obj.pump_filter_on,
            "cancellation_on": obj.cancellation_on,
            "cancellation_phase": _unstructure_basic_or_parameter_model(
                obj.cancellation_phase
            ),
            "cancellation_attenuation": _unstructure_basic_or_parameter_model(
                obj.cancellation_attenuation
            ),
            "cancellation_source": _converter.unstructure(
                obj.cancellation_source, CancellationSourceModel
            ),
            "cancellation_source_frequency": obj.cancellation_source_frequency,
            "alc_on": obj.alc_on,
            "probe_on": obj.probe_on,
            "probe_frequency": _unstructure_basic_or_parameter_model(
                obj.probe_frequency
            ),
            "probe_power": _unstructure_basic_or_parameter_model(obj.probe_power),
        }

    @classmethod
    def _structure(cls, d, _):
        pump_frequency = _structure_basic_or_parameter_model(d["pump_frequency"])
        pump_power = _structure_basic_or_parameter_model(d["pump_power"])
        cancellation_phase = _structure_basic_or_parameter_model(
            d["cancellation_phase"]
        )
        cancellation_attenuation = _structure_basic_or_parameter_model(
            d["cancellation_attenuation"]
        )
        probe_frequency = _structure_basic_or_parameter_model(d["probe_frequency"])
        probe_power = _structure_basic_or_parameter_model(d["probe_power"])
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
                obj.amplitude_scaling
            ),
            "phase_shift": _unstructure_basic_or_parameter_model(obj.phase_shift),
        }

    @classmethod
    def _structure(cls, d, _):
        return cls._target_class(
            source=d["source"],
            amplitude_scaling=_structure_basic_or_parameter_model(
                d["amplitude_scaling"]
            ),
            phase_shift=_structure_basic_or_parameter_model(d["phase_shift"]),
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
        voltage_offset = _structure_basic_or_parameter_model(d["voltage_offset"])
        range = d["range"]
        threshold = d["threshold"]
        amplitude = _structure_basic_or_parameter_model(d["amplitude"])
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
class DataServerModel:
    uid: str
    host: str
    port: str | int
    api_level: int
    _target_class: ClassVar[Type] = DataServer

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "host": obj.host,
            "port": obj.port,
            "api_level": obj.api_level,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            host=obj["host"],
            port=obj["port"],
            api_level=obj["api_level"],
        )


@attrs.define
class ConnectionModel:
    direction: IODirectionModel
    local_path: str | None
    local_port: str | None
    remote_path: str | None
    remote_port: str | None
    signal_type: IOSignalTypeModel | None
    _target_class: ClassVar[Type] = Connection

    @classmethod
    def _unstructure(cls, obj):
        return {
            "direction": _converter.unstructure(
                obj.direction, Union[IODirectionModel, None]
            ),
            "local_path": obj.local_path,
            "local_port": obj.local_port,
            "remote_path": obj.remote_path,
            "remote_port": obj.remote_port,
            "signal_type": _converter.unstructure(
                obj.signal_type, Union[IOSignalTypeModel, None]
            ),
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            direction=None
            if obj["direction"] is None
            else IODirectionModel._target_class.value(obj["direction"]),
            local_path=obj["local_path"],
            local_port=obj["local_port"],
            remote_path=obj["remote_path"],
            remote_port=obj["remote_port"],
            signal_type=None
            if obj["signal_type"] is None
            else IOSignalTypeModel._target_class.value(obj["signal_type"]),
        )


@attrs.define
class ZIStandardInstrumentModel:
    uid: str
    interface: str
    connections: list[ConnectionModel]
    server_uid: str | None
    address: str | None
    device_options: str | None
    reference_clock_source: ReferenceClockSourceModel | str | None
    _target_class: ClassVar[Type] = ZIStandardInstrument
    _instrument_map: ClassVar[dict] = {
        "HDAWG": HDAWG,
        "UHFQA": UHFQA,
        "SHFQC": SHFQC,
        "SHFSG": SHFSG,
        "SHFPPC": SHFPPC,
        "SHFQA": SHFQA,
        "PQSC": PQSC,
    }

    @classmethod
    def _unstructure(cls, obj):
        if type(obj).__name__ not in cls._instrument_map:
            raise ValueError(
                f"Unsupported instrument type: {type(obj).__name__} when unstructuring ZIStandardInstrumentModel"
            )
        return {
            "uid": obj.uid,
            "interface": obj.interface,
            "connections": [
                _converter.unstructure(i, ConnectionModel) for i in obj.connections
            ],
            "server_uid": obj.server_uid,
            "address": obj.address,
            "device_options": obj.device_options,
            "reference_clock_source": _converter.unstructure(
                obj.reference_clock_source, Union[ReferenceClockSourceModel, str, None]
            ),
            "_instrument_type": type(obj).__name__,
        }

    @classmethod
    def _structure(cls, obj, _):
        if obj["_instrument_type"] not in cls._instrument_map:
            raise ValueError(
                f"Unsupported instrument type: {obj['_instrument_type']} when structuring InstrumentModel"
            )
        instrument = cls._instrument_map[obj["_instrument_type"]]
        if obj["reference_clock_source"] is None or isinstance(
            obj["reference_clock_source"], str
        ):
            ref_clk_source = obj["reference_clock_source"]
        else:
            ref_clk_source = ReferenceClockSourceModel._target_class.value(
                obj["reference_clock_source"]
            )
        return instrument(
            uid=obj["uid"],
            interface=obj["interface"],
            connections=[
                _converter.structure(i, ConnectionModel) for i in obj["connections"]
            ],
            server_uid=obj["server_uid"],
            address=obj["address"],
            device_options=obj["device_options"],
            reference_clock_source=ref_clk_source,
        )


@attrs.define
class PhysicalChannelModel:
    uid: str
    name: str | None
    type: PhysicalChannelTypeModel | None
    path: str | None
    calibration: SignalCalibrationModel | None
    _target_class: ClassVar[Type] = PhysicalChannel

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "name": obj.name,
            "type": _converter.unstructure(obj.type, Union[PhysicalChannelType, None]),
            "path": obj.path,
            "calibration": _converter.unstructure(
                obj._calibration, Union[SignalCalibrationModel, None]
            ),
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            name=obj["name"],
            type=None
            if obj["type"] is None
            else PhysicalChannelTypeModel._target_class.value(obj["type"]),
            path=obj["path"],
            calibration=None
            if obj["calibration"] is None
            else _converter.structure(obj["calibration"], SignalCalibrationModel),
        )


@attrs.define
class PhysicalChannelGroupModel:
    uid: str
    channels: dict[str, PhysicalChannelModel]
    _target_class: ClassVar[Type] = PhysicalChannelGroup

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "channels": {
                k: _converter.unstructure(v, PhysicalChannelModel)
                for k, v in obj.channels.items()
            },
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            channels={
                k: _converter.structure(v, PhysicalChannelModel)
                for k, v in obj["channels"].items()
            },
        )


@attrs.define
class LogicalSignalModel:
    uid: str
    name: str | None
    calibration: SignalCalibration | None
    physical_channel: PhysicalChannelModel | None
    path: str | None
    direction: IODirectionModel | None
    _target_class: ClassVar[Type] = LogicalSignal

    @classmethod
    def _unstructure(cls, obj):
        if obj.calibration:
            calibration = _converter.unstructure(
                obj.calibration, SignalCalibrationModel
            )
        else:
            calibration = None
        if obj.physical_channel:
            physical_channel = _converter.unstructure(
                obj.physical_channel, PhysicalChannelModel
            )
        else:
            physical_channel = None
        return {
            "uid": obj.uid,
            "name": obj.name,
            "calibration": calibration,
            "physical_channel": physical_channel,
            "path": obj.path,
            "direction": _converter.unstructure(obj.direction, IODirectionModel)
            if obj.direction
            else None,
        }

    @classmethod
    def _structure(cls, obj, _):
        if obj["calibration"]:
            calibration = _converter.structure(
                obj["calibration"], SignalCalibrationModel
            )
        else:
            calibration = None
        if obj["physical_channel"]:
            physical_channel = _converter.structure(
                obj["physical_channel"], PhysicalChannelModel
            )
        else:
            physical_channel = None
        return cls._target_class(
            uid=obj["uid"],
            name=obj["name"],
            calibration=calibration,
            physical_channel=physical_channel,
            path=obj["path"],
            direction=IODirectionModel._target_class.value(obj["direction"])
            if obj["direction"]
            else None,
        )


@attrs.define
class LogicalSignalGroupModel:
    uid: str | None
    logical_signals: dict[str, LogicalSignalModel]
    _target_class: ClassVar[Type] = LogicalSignalGroup

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "logical_signals": {
                k: _converter.unstructure(v, LogicalSignalModel)
                for k, v in obj.logical_signals.items()
            },
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            logical_signals={
                k: _converter.structure(v, LogicalSignalModel)
                for k, v in obj["logical_signals"].items()
            },
        )


# Top-level models


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


# Supporting functions for serialization


def _register_models(converter, models: Iterable):
    # Use predicate hooks to avoid type evaluation problems
    # for complicate type in Python 3.9.
    # TODO: When dropping support for Python 3.9,
    # Remove registering predicate hooks (register_(un)structure_hook_func)
    # and use dispatcher hooks instead (register_(un)structure_hook)
    # See the docstring of this module for more details.
    def _predicate(cls):
        return lambda obj: obj is cls

    for model in models:
        if hasattr(model, "_unstructure"):
            converter.register_unstructure_hook_func(
                _predicate(model), model._unstructure
            )
        else:
            converter.register_unstructure_hook(
                model, make_dict_unstructure_fn(model, converter)
            )
        if hasattr(model, "_structure"):
            converter.register_structure_hook_func(_predicate(model), model._structure)
        else:
            converter.register_structure_hook(
                model, make_dict_structure_fn(model, converter)
            )


def _collect_models() -> frozenset:
    """Collect all attrs models for serialization. Enum models are
    already registered in the pre-configured converter."""
    current_module = sys.modules[__name__]
    subclasses = []
    for _, cls in inspect.getmembers(current_module):
        if attrs.has(cls) and hasattr(cls, "_target_class"):
            subclasses.append(cls)

    return frozenset(subclasses)


def make_laboneq_converter() -> Converter:
    converter = make_converter()
    # Collect all attrs models for serialization
    _register_models(converter, _collect_models())
    converter.register_structure_hook(
        ArrayLike_Model, lambda obj, cls: numpy.asarray(obj)
    )
    converter.register_unstructure_hook(
        ArrayLike_Model,
        lambda obj: obj.tolist() if isinstance(obj, numpy.ndarray) else obj,
    )

    converter.register_unstructure_hook(
        ParameterModel,
        _unstructure_parameter_model,
    )
    converter.register_structure_hook(
        ParameterModel,
        _structure_parameter_model,
    )

    return converter


# used internally for recursive serialization
_converter = make_laboneq_converter()
