# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

import sys
from functools import partial
from typing import ClassVar, Type, Union

import attrs

from laboneq.dsl.experiment.play_pulse import PlayPulse
from laboneq.dsl.experiment.pulse import PulseFunctional, PulseSampled
from laboneq.serializers._cache import PulseSampledCache

from ._calibration import (
    ParameterModel,
    _structure_basic_or_parameter_model,
    _unstructure_basic_or_parameter_model,
)
from ._calibration import (
    make_converter as make_converter_calibration,
)
from ._common import (
    ArrayLike_Model,
    collect_models,
    register_models,
    structure_union_generic_type,
    unstructure_union_generic_type,
)


@attrs.define
class PulseSampledModel:
    samples: ArrayLike_Model
    uid: str
    can_compress: bool
    _target_class: ClassVar[Type] = PulseSampled

    @classmethod
    def _unstructure(cls, obj):
        _cache = PulseSampledCache.get_pulse_cache()
        if id(obj) not in _cache and isinstance(obj, PulseSampled):
            ref = _cache.add(obj)
            return {
                "samples": _converter.unstructure(obj.samples, ArrayLike_Model),
                "uid": obj.uid,
                "can_compress": obj.can_compress,
                "$ref": ref,
            }
        else:
            return {"$ref": ref}

    @classmethod
    def _structure(cls, obj, _):
        # Here it's assumed that the original object was always serialized
        # before its references. This is already made sure by the serialization,
        # in which no reordering is done after converting the objects to dicts.
        # which is also a sequential process.
        _cache = PulseSampledCache.get_pulse_cache()
        if obj.get("$ref") is None:
            de = cls._target_class(
                samples=_converter.structure(obj["samples"], ArrayLike_Model),
                uid=obj["uid"],
                can_compress=obj["can_compress"],
            )
            _cache.add(de)
            return de
        else:
            return _cache.get(obj["$ref"])


@attrs.define
class PulseFunctionalModel:
    function: str
    uid: str
    amplitude: float
    length: float
    can_compress: bool
    pulse_parameters: PulseParameterModel
    _target_class: ClassVar[Type] = PulseFunctional

    # simple enough to not require customized unstructure
    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            function=obj["function"],
            uid=obj["uid"],
            amplitude=obj["amplitude"],
            length=obj["length"],
            can_compress=obj["can_compress"],
            pulse_parameters=obj["pulse_parameters"],
        )


PulseModel = Union[PulseSampledModel, PulseFunctionalModel]


def _unstructure_pulse_model(obj, _converter):
    return unstructure_union_generic_type(
        obj, [PulseSampledModel, PulseFunctionalModel], _converter
    )


def _structure_pulse_model(d, _, _converter):
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(
        d, [PulseSampledModel, PulseFunctionalModel], _converter
    )


PulseParameterModel = dict[
    str, Union[float, int, str, bool, complex]
]  # assume to be a dict with simple types

# Operation-related models


@attrs.define
class PlayPulseModel:
    signal: str
    pulse: PulseModel
    amplitude: Union[float, complex, ParameterModel]
    increment_oscillator_phase: Union[float, ParameterModel]
    phase: float
    set_oscillator_phase: float
    length: Union[float, ParameterModel]
    pulse_parameters: PulseParameterModel | None
    precompensation_clear: bool | None
    marker: dict | None
    _target_class: ClassVar[Type] = PlayPulse

    @classmethod
    def _unstructure(cls, obj):
        return {
            "signal": obj.signal,
            "pulse": _converter.unstructure(obj.pulse, PulseModel),
            "amplitude": _unstructure_basic_or_parameter_model(
                obj.amplitude, _converter
            ),
            "increment_oscillator_phase": _unstructure_basic_or_parameter_model(
                obj.increment_oscillator_phase, _converter
            ),
            "phase": obj.phase,
            "set_oscillator_phase": obj.set_oscillator_phase,
            "length": _unstructure_basic_or_parameter_model(obj.length, _converter),
            "pulse_parameters": obj.pulse_parameters,
            "precompensation_clear": obj.precompensation_clear,
            "marker": obj.marker,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            signal=obj["signal"],
            pulse=_converter.structure(obj["pulse"], PulseModel),
            amplitude=_structure_basic_or_parameter_model(obj["amplitude"], _converter),
            increment_oscillator_phase=_structure_basic_or_parameter_model(
                obj["increment_oscillator_phase"], _converter
            ),
            phase=obj["phase"],
            set_oscillator_phase=obj["set_oscillator_phase"],
            length=_structure_basic_or_parameter_model(obj["length"], _converter),
            pulse_parameters=obj["pulse_parameters"],
            precompensation_clear=obj["precompensation_clear"],
            marker=obj["marker"],
        )


def make_converter():
    _converter = make_converter_calibration()
    _converter.register_unstructure_hook(
        PulseModel,
        partial(_unstructure_pulse_model, _converter=_converter),
    )
    _converter.register_structure_hook(
        PulseModel,
        partial(_structure_pulse_model, _converter=_converter),
    )
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
