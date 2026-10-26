# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from enum import Enum
from functools import partial
from types import NoneType
from typing import Callable, ClassVar, NewType, Optional, Type, Union

import attrs
import numpy
from cattrs import Converter

from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.types.enums.dsl_version import DSLVersion
from laboneq.core.types.enums.execution_type import ExecutionType
from laboneq.core.types.enums.repetition_mode import RepetitionMode
from laboneq.core.types.enums.section_alignment import SectionAlignment
from laboneq.core.types.enums.section_timing_mode import SectionTimingMode
from laboneq.dsl.coprocessor.operations import _Ensure, _MarkStale, _Send
from laboneq.dsl.coprocessor.predicate import _IsLive
from laboneq.dsl.experiment.acquire import Acquire
from laboneq.dsl.experiment.call import Call
from laboneq.dsl.experiment.delay import Delay
from laboneq.dsl.experiment.do_until import DoUntilSection
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.experiment.play_pulse import PlayPulse
from laboneq.dsl.experiment.pulse import PulseFunctional, PulseSampled
from laboneq.dsl.experiment.reserve import Reserve
from laboneq.dsl.experiment.reset_oscillator_phase import ResetOscillatorPhase
from laboneq.dsl.experiment.section import (
    AcquireLoopRt,
    Case,
    Match,
    PRNGLoop,
    PRNGSetup,
    Section,
    Sweep,
)
from laboneq.dsl.experiment.set_node import SetNode
from laboneq.dsl.parameter import LinearSweepParameter, SweepParameter
from laboneq.dsl.prng import PRNG, PRNGSample
from laboneq.serializers._cache import PulseCache, SectionCache
from laboneq.serializers.implementations._models import _coprocessor
from laboneq.serializers.implementations._models._coprocessor import (
    StreamModel,
    VariableModel,
)

from ._calibration import (
    LinearSweepParameterModel,
    ParameterModel,
    SignalCalibrationModel,
    SweepParameterModel,
    structure_basic_or_parameter_model,
    unstructure_basic_or_parameter_model,
)
from ._calibration import (
    make_converter as make_converter_calibration,
)
from ._common import (
    ArrayLike_Model,
    BytesModel,
    ComplexModel,
    collect_models,
    register_models,
    structure_union_generic_type,
    unstructure_union_generic_type,
)

# Enums type


class ExecutionTypeModel(Enum):
    NEAR_TIME = "controller"
    REAL_TIME = "hardware"
    _target_class = ExecutionType


class SectionAlignmentModel(Enum):
    LEFT = "left"
    RIGHT = "right"
    _target_class = SectionAlignment


class SectionTimingModeModel(Enum):
    RELAXED = "relaxed"
    STRICT = "strict"
    _target_class = SectionTimingMode


class AveragingModeModel(Enum):
    SEQUENTIAL = "sequential"
    CYCLIC = "cyclic"
    SINGLE_SHOT = "single_shot"
    _target_class = AveragingMode


class RepetitionModeModel(Enum):
    FASTEST = "fastest"
    CONSTANT = "constant"
    AUTO = "auto"
    _target_class = RepetitionMode


class AcquisitionTypeModel(Enum):
    INTEGRATION = "integration_trigger"
    SPECTROSCOPY_IQ = "spectroscopy"
    SPECTROSCOPY_PSD = "spectroscopy_psd"
    SPECTROSCOPY = SPECTROSCOPY_IQ
    DISCRIMINATION = "discrimination"
    RAW = "RAW"
    _target_class = AcquisitionType


class DSLVersionModel(Enum):
    ALPHA = None
    V2_4_0 = "2.4.0"
    V2_5_0 = "2.5.0"
    V3_0_0 = "3.0.0"
    LATEST = "3.0.0"
    _target_class = DSLVersion


@PulseCache.cache
@attrs.define
class PulseSampledModel:
    samples: ArrayLike_Model
    uid: str
    can_compress: bool
    _target_class: ClassVar[Type] = PulseSampled

    @classmethod
    def _unstructure(cls, obj):
        return {
            "samples": _converter.unstructure(obj.samples, ArrayLike_Model),
            "uid": obj.uid,
            "can_compress": obj.can_compress,
        }

    @classmethod
    def _structure(cls, obj, _):
        pulse = cls._target_class(
            samples=_converter.structure(obj["samples"], ArrayLike_Model),
            uid=obj["uid"],
            can_compress=obj["can_compress"],
        )
        return pulse


@PulseCache.cache
@attrs.define
class PulseFunctionalModel:
    function: str
    uid: str
    amplitude: ParameterModel | float | complex | numpy.number | None
    length: ParameterModel | float | None
    can_compress: bool
    pulse_parameters: PulseParameterModel | None
    _target_class: ClassVar[Type] = PulseFunctional

    @classmethod
    def _unstructure(cls, obj):
        return {
            "function": obj.function,
            "uid": obj.uid,
            "amplitude": unstructure_basic_or_parameter_model(
                obj.amplitude, _converter
            ),
            "length": unstructure_basic_or_parameter_model(obj.length, _converter),
            "can_compress": obj.can_compress,
            "pulse_parameters": None
            if obj.pulse_parameters is None
            else _unstructure_pulse_parameter_model(obj.pulse_parameters),
        }

    @classmethod
    def _structure(cls, obj, _):
        # cattrs refuses to guess when structuring complex | numpy.number | ArbitraryModel
        # so we have to do it manually.
        return cls._target_class(
            function=obj["function"],
            uid=obj["uid"],
            amplitude=structure_basic_or_parameter_model(obj["amplitude"], _converter),
            length=structure_basic_or_parameter_model(obj["length"], _converter),
            can_compress=obj["can_compress"],
            pulse_parameters=None
            if obj["pulse_parameters"] is None
            else _structure_pulse_parameter_model(obj["pulse_parameters"]),
        )


PulseModel = PulseSampledModel | PulseFunctionalModel


def _unstructure_pulse_model(obj, _converter):
    return unstructure_union_generic_type(
        obj, [PulseSampledModel, PulseFunctionalModel], _converter
    )


def _structure_pulse_model(d, _, _converter):
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(
        d, [PulseSampledModel, PulseFunctionalModel], _converter
    )


PulseParameterValueBasicTypes = (
    # Pulse parameter value types that are stored using the
    # basic structure and unstructure functions from _common:
    int,
    float,
    str,
    bool,
    type(None),
)


PulseParameterValueModels = (
    # The models corresponding to the types in PulseParameterValueModelTypes:
    BytesModel,
    ComplexModel,
    SweepParameterModel,
    LinearSweepParameterModel,
)

PulseParameterValueModelTypes = tuple(
    model._target_class for model in PulseParameterValueModels
)

PulseParameterValueModel = (
    Union[PulseParameterValueBasicTypes]
    | Union[PulseParameterValueModelTypes]
    | list["PulseParameterValueModel"]
    | dict[str, "PulseParameterValueModel"]
)

PulseParameterModel = dict[str, PulseParameterValueModel]


_UNSUPPORTED_VALUE_HINT = (
    "Supported types are int, float, complex, str, bytes, bool, None, SweepParameter,"
    " LinearSweepParameter, and (nested) lists or str-keyed dicts of these; a sweep"
    " parameter must be the value itself, as it is not resolved when nested."
    " Convert the value to a supported type (numpy arrays: `.tolist()`), or serialize it"
    " to `bytes` and deserialize it inside the pulse sampler or near-time callback that"
    " consumes it."
)


def _unsupported_value_error(value) -> ValueError:
    """Build the error raised for a value LabOne Q cannot serialize.

    Callers may prefix the message with the origin of the value (which pulse parameter or
    callback argument it came from); the message itself therefore names no origin.
    """
    return ValueError(
        f"Unsupported value of type {type(value).__name__}: {value!r}."
        f" {_UNSUPPORTED_VALUE_HINT}"
    )


def _unstructure_pulse_parameter_model(obj):
    return {k: _unstructure_pulse_parameter_value(v) for k, v in obj.items()}


def _unstructure_pulse_parameter_value(value):
    if isinstance(value, PulseParameterValueBasicTypes):
        return value
    if isinstance(value, list):
        return [_unstructure_pulse_parameter_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _unstructure_pulse_parameter_value(v) for k, v in value.items()}
    if isinstance(value, PulseParameterValueModelTypes):
        return unstructure_union_generic_type(
            value,
            PulseParameterValueModels,
            _converter,
        )
    raise _unsupported_value_error(value)


def _structure_pulse_parameter_model(obj):
    return {k: _structure_pulse_parameter_value(v) for k, v in obj.items()}


def _structure_pulse_parameter_value(value):
    if isinstance(value, PulseParameterValueBasicTypes):
        return value
    if isinstance(value, dict) and "_type" in value:
        return structure_union_generic_type(
            value,
            PulseParameterValueModels,
            _converter,
        )
    if isinstance(value, list):
        return [_structure_pulse_parameter_value(item) for item in value]
    if isinstance(value, dict):
        return {k: _structure_pulse_parameter_value(v) for k, v in value.items()}
    raise _unsupported_value_error(value)


def _guess_sweep_parameter_type_v4_to_v5(obj):
    """Adds an appropriate _type entry to an unstructured Experiment v4 sweep parameter.

    Arguments:
        obj:
            Unstructured Experiment v4 sweep parameter data.

    Returns:
        Experiment v5 sweep parameter data with an appropriate `_type` key added.

    Raises:
        ValueError:
            When `obj` does not appear to be a valid sweep.
    """
    if "_type" in obj:
        pass
    elif obj.keys() == {"uid", "start", "stop", "count", "axis_name"}:
        obj = obj | {"_type": "LinearSweepParameter"}
    elif obj.keys() == {"uid", "values", "axis_name", "driven_by"}:
        obj = obj | {"_type": "SweepParameter"}
    else:
        raise ValueError(
            f"Unsupported sweep parameter type encountered while converting Experiment v4 to v5: {obj!r}"
        )
    return obj


def _convert_pulse_parameter_model_v4_to_v5(obj):
    """Convert unstructured pulse parameter model data from Experiment v4 to v5.

    Arguments:
        obj:
            Unstructure Experiment v4 pulse parameter model data.

    Returns:
        Experiment v5 pulse parameter model data.

    Raises:
        ValueError:
            When an unsupported type is encountered.
    """
    # In version 4, the PulseParameterModel supported only the following type signature
    # for pulse parameters:
    #
    #  dict[
    #    str, Union[float, int, str, bool, complex, ParameterModel, list[ParameterModel]]
    #  ]
    #
    # The behaviour was as follows:
    #
    # - float, int, str and bool were stored the same way as in v5.
    # - None was not officially supported but was stored the same way as in v5.
    # - complex were stored as a list but raised `ValueError: Only input mappings are supported` when deserializing.
    # - dict was not officially supported but were stored and raised `ValueError: Couldn't disambiguate` when deserializing.
    # - list was not officially supported except for lists of `ParameterModel` but were stored as a list and
    #   raised `ValueError: Only input mappings are supported` when deserializing.
    # - SweepParameter and LinearSweepParameter were both stored as dict without an `_type`. They were disambiguated by
    #   cattrs magic that looked at the attributes.
    # - unsupported values did raise when unstructuring in v4. Instead they placed the unsupported object directly in the
    #   unstructured dictionary and it likely errored later when converting the dict to JSON.
    #
    # This converter deals with all of the above using the following logic:
    #
    # - float, int, str, bool and None values are kept as-is
    # - dict values are assumed to be SweepParameters or LinearSweepParameters
    # - lists are assumed to contain only SweepParameters or LinearSweepParameters
    # - unstructued SweepParameters and LinearSweepParameters are disambiguated using
    #   the names of the keys they contain.
    # - other values raise a ValueError.
    ret = {}
    for k, v in obj.items():
        if isinstance(
            v,
            (
                float,
                int,
                str,
                bool,
                NoneType,
            ),
        ):
            # float, int, str, bool, None
            ret[k] = v
        elif isinstance(v, dict):
            # ParameterModel (SweepParameter or LinearSweepParameter)
            ret[k] = _guess_sweep_parameter_type_v4_to_v5(v)
        elif isinstance(v, list) and all(isinstance(item, dict) for item in v):
            # list[ParameterModel] (list of SweepParameter or LinearSweepParameter)
            ret[k] = [_guess_sweep_parameter_type_v4_to_v5(item) for item in v]
        else:
            raise ValueError(
                f"Unsupported parameter value encountered while converting Experiment v4 to v5: key={k!r}, value={v!r}"
            )
    return ret


# Operation-related models
@attrs.define
class PlayPulseModel:
    signal: str | None
    pulse: PulseModel | None
    amplitude: float | numpy.number | ParameterModel | None
    increment_oscillator_phase: float | ParameterModel | None
    phase: float | ParameterModel | None
    set_oscillator_phase: float | None
    length: float | ParameterModel | None
    pulse_parameters: PulseParameterModel | None
    precompensation_clear: bool | None
    marker: dict | None
    _target_class: ClassVar[Type] = PlayPulse

    @classmethod
    def _unstructure(cls, obj):
        if obj.pulse is None:
            pulse = None
        else:
            pulse = _converter.unstructure(obj.pulse, PulseModel)
        return {
            "signal": obj.signal,
            "pulse": pulse,
            "amplitude": unstructure_basic_or_parameter_model(
                obj.amplitude, _converter
            ),
            "increment_oscillator_phase": unstructure_basic_or_parameter_model(
                obj.increment_oscillator_phase, _converter
            ),
            "phase": unstructure_basic_or_parameter_model(obj.phase, _converter),
            "set_oscillator_phase": obj.set_oscillator_phase,
            "length": unstructure_basic_or_parameter_model(obj.length, _converter),
            "pulse_parameters": None
            if obj.pulse_parameters is None
            else _unstructure_pulse_parameter_model(obj.pulse_parameters),
            "precompensation_clear": obj.precompensation_clear,
            "marker": obj.marker,
        }

    @classmethod
    def _structure(cls, obj, _):
        # cattrs refuses to guess when structuring complex | numpy.number | ArbitraryModel
        # so we have to do it manually.
        if obj["pulse"] is None:
            pulse = None
        else:
            pulse = _converter.structure(obj["pulse"], PulseModel)
        return cls._target_class(
            signal=obj["signal"],
            pulse=pulse,
            amplitude=structure_basic_or_parameter_model(obj["amplitude"], _converter),
            increment_oscillator_phase=structure_basic_or_parameter_model(
                obj["increment_oscillator_phase"], _converter
            ),
            phase=structure_basic_or_parameter_model(obj["phase"], _converter),
            set_oscillator_phase=obj["set_oscillator_phase"],
            length=structure_basic_or_parameter_model(obj["length"], _converter),
            pulse_parameters=None
            if obj["pulse_parameters"] is None
            else _structure_pulse_parameter_model(obj["pulse_parameters"]),
            precompensation_clear=obj["precompensation_clear"],
            marker=obj["marker"],
        )


@attrs.define
class ReserveModel:
    signal: str
    _target_class: ClassVar[Type] = Reserve


@attrs.define
class ResetOscillatorPhaseModel:
    signal: str | None
    _target_class: ClassVar[Type] = ResetOscillatorPhase


@attrs.define
class SetNodeModel:
    path: str
    # Assume that the value is a simple type
    value: Optional[Union[str, int, float, complex, bool]]
    _target_class: ClassVar[Type] = SetNode


@attrs.define
class DelayModel:
    signal: str | None
    time: float | ParameterModel | None
    precompensation_clear: bool | None
    _target_class: ClassVar[Type] = Delay


@attrs.define
class AcquireModel:
    signal: str | None
    handle: str | None
    kernel: PulseModel | list[PulseModel] | None
    length: float | None
    pulse_parameters: PulseParameterModel | list[PulseParameterModel] | None
    _target_class: ClassVar[Type] = Acquire

    @classmethod
    def _unstructure(cls, obj):
        if isinstance(obj.kernel, list):
            kernel = [_converter.unstructure(k, PulseModel) for k in obj.kernel]
        elif obj.kernel is None:
            kernel = None
        else:
            kernel = _converter.unstructure(obj.kernel, PulseModel)

        if isinstance(obj.pulse_parameters, list):
            pulse_parameters = [
                _unstructure_pulse_parameter_model(p) for p in obj.pulse_parameters
            ]
        elif obj.pulse_parameters is None:
            pulse_parameters = None
        else:
            pulse_parameters = _unstructure_pulse_parameter_model(obj.pulse_parameters)

        return {
            "signal": obj.signal,
            "handle": obj.handle,
            "kernel": kernel,
            "length": obj.length,
            "pulse_parameters": pulse_parameters,
        }

    @classmethod
    def _structure(cls, obj, _):
        if isinstance(obj["kernel"], list):
            kernel = [
                _converter.structure(k, PulseModel) if k is not None else k
                for k in obj["kernel"]
            ]
        else:
            kernel = (
                _converter.structure(obj["kernel"], PulseModel)
                if obj["kernel"]
                else None
            )

        if isinstance(obj["pulse_parameters"], list):
            pulse_parameters = [
                _structure_pulse_parameter_model(p) for p in obj["pulse_parameters"]
            ]
        elif obj["pulse_parameters"] is None:
            pulse_parameters = None
        else:
            pulse_parameters = _structure_pulse_parameter_model(obj["pulse_parameters"])
        return cls._target_class(
            signal=obj["signal"],
            handle=obj["handle"],
            kernel=kernel,
            length=obj["length"],
            pulse_parameters=pulse_parameters,
        )


@attrs.define
class CallModel:
    func_name: str | Callable
    # args could be anything, here we limit it to simple types.
    args: dict[
        str,
        Union[ParameterModel | str | int | float | bool | complex | None],
    ]
    _target_class: ClassVar[Type] = Call

    @classmethod
    def _unstructure(cls, obj):
        args = {}
        for k, v in obj.args.items():
            if isinstance(v, (SweepParameter, LinearSweepParameter)):
                args[k] = _converter.unstructure(v, ParameterModel)
            else:
                args[k] = v
        if callable(obj.func_name):
            func_name = obj.func_name.__name__
        else:
            func_name = obj.func_name
        return {
            "func_name": func_name,
            "args": args,
        }

    @classmethod
    def _structure(cls, obj, _):
        args = {}
        for k, v in obj["args"].items():
            if isinstance(v, dict):
                args[k] = _converter.structure(v, ParameterModel)
            else:
                args[k] = v
        return cls._target_class(
            func_name=obj["func_name"],
            **args,
        )


def _unstructure_hqcs_target(target):
    """Serialize a `Variable | Pulse | None` cross-reference.

    Both arms resolve to a `$ref` via their identity cache once the target has
    been serialized at its canonical site (a stream field accessor for a
    Variable; a stream field or a section for a Pulse). The `_kind` tag selects
    the model on the way back in.
    """
    if target is None:
        return None
    from laboneq.dsl.variable import Variable

    if isinstance(target, Variable):
        return {"_kind": "variable", **_converter.unstructure(target, VariableModel)}
    return {"_kind": "pulse", **_converter.unstructure(target, PulseModel)}


def _structure_hqcs_target(d):
    if d is None:
        return None
    if d["_kind"] == "variable":
        return _converter.structure(d, VariableModel)
    return _converter.structure(d, PulseModel)


@attrs.define
class _SendModel:
    """Model for `_Send` operations (`send(stream, **kwargs)` in a section).

    The stream is cross-referenced by `$ref` into the experiment's stream
    identity cache (`StreamCache`); the canonical serialization lives in the
    experiment's `streams` list, which is serialized before the sections.
    """

    stream: StreamModel
    literal_kwargs: dict
    _target_class: ClassVar[Type] = _Send

    @classmethod
    def _unstructure(cls, obj):
        return {
            "stream": _converter.unstructure(obj.stream, StreamModel),
            "literal_kwargs": obj.literal_kwargs,
        }

    @classmethod
    def _structure(cls, obj, _):
        stream = _converter.structure(obj["stream"], StreamModel)
        return _Send(stream=stream, literal_kwargs=obj.get("literal_kwargs", {}))


@attrs.define
class _MarkStaleModel:
    """Model for `_MarkStale` operations (`mark_stale(target)` in a section).

    The target is a `Variable` or a `Pulse`, cross-referenced by `$ref` via
    its identity cache.
    """

    target: dict | None
    _target_class: ClassVar[Type] = _MarkStale

    @classmethod
    def _unstructure(cls, obj):
        return {"target": _unstructure_hqcs_target(obj.target)}

    @classmethod
    def _structure(cls, obj, _):
        return _MarkStale(target=_structure_hqcs_target(obj.get("target")))


@attrs.define
class _EnsureModel:
    """Model for `_Ensure` operations (`ensure(condition)` in a section).

    The condition is an `is_live(...)` predicate whose target is
    cross-referenced by `$ref` via its identity cache.
    """

    condition: dict | None
    _target_class: ClassVar[Type] = _Ensure

    @classmethod
    def _unstructure(cls, obj):
        return {"condition": _unstructure_hqcs_condition(obj.condition)}

    @classmethod
    def _structure(cls, obj, _):
        condition = _structure_hqcs_condition(obj.get("condition"))
        assert condition is not None
        return _Ensure(condition=condition)


_operation_types = [
    AcquireModel,
    CallModel,
    DelayModel,
    _EnsureModel,
    _MarkStaleModel,
    PlayPulseModel,
    ReserveModel,
    SetNodeModel,
    _SendModel,
    ResetOscillatorPhaseModel,
]
_operation_types_target_class = {cl._target_class.__name__ for cl in _operation_types}
OperationModel = Union[
    AcquireModel,
    CallModel,
    DelayModel,
    _EnsureModel,
    _MarkStaleModel,
    PlayPulseModel,
    ReserveModel,
    SetNodeModel,
    _SendModel,
    ResetOscillatorPhaseModel,
]


def _unstructure_operation_model(obj, _converter: Converter):
    return unstructure_union_generic_type(obj, _operation_types, _converter)


def _structure_operation_model(d, _, _converter: Converter):
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(d, _operation_types, _converter)


PlayAfterModel = NewType("PlayAfterModel", str | Section | list[str | Section] | None)

MatchVariableModel = NewType("MatchVariableModel", object)


def _unstructure_play_after_model(obj):
    if isinstance(obj, list):
        play_after = [p.uid for p in obj]
    elif obj is None or isinstance(obj, str):
        play_after = obj
    else:
        play_after = obj.uid
    return play_after


def _structure_play_after_model(obj, _):
    if isinstance(obj, list):
        play_after = [
            p if isinstance(p, str) else _converter.structure(p, SectionModel)
            for p in obj
        ]
    elif obj is None or isinstance(obj, str):
        play_after = obj
    else:
        play_after = _converter.structure(obj, SectionModel) if obj else None
    return play_after


@SectionCache.cache
@attrs.define
class SectionModel:
    uid: str | None
    name: str
    alignment: SectionAlignmentModel
    execution_type: ExecutionType | None
    length: float | None
    play_after: PlayAfterModel
    children: list[AllSectionModel | OperationModel]
    trigger: dict[str, dict[str, int]]
    section_timing_mode: SectionTimingModeModel | None

    on_system_grid: bool | None
    _target_class: ClassVar[Type] = Section


@attrs.define
class MatchModel(SectionModel):
    handle: str | None
    user_register: int | None
    prng_sample: PRNGSampleModel | None
    sweep_parameter: ParameterModel | None
    local: bool | None
    variable: MatchVariableModel = None
    _target_class: ClassVar[Type] = Match


@attrs.define
class AcquireLoopRtModel(SectionModel):
    acquisition_type: AcquisitionTypeModel
    averaging_mode: AveragingModeModel
    count: int | None
    execution_type: ExecutionTypeModel
    repetition_mode: RepetitionModeModel
    repetition_time: float | None
    reset_oscillator_phase: bool
    _target_class: ClassVar[Type] = AcquireLoopRt


@attrs.define
class SweepModel(SectionModel):
    parameters: ParameterModel | list[ParameterModel]
    reset_oscillator_phase: bool
    chunk_count: int
    auto_chunking: bool
    _target_class: ClassVar[Type] = Sweep


@attrs.define
class CaseModel(SectionModel):
    state: int
    _target_class: ClassVar[Type] = Case


@attrs.define
class PRNGSetupModel(SectionModel):
    prng: PRNGModel | None
    _target_class: ClassVar[Type] = PRNGSetup


@attrs.define
class PRNGLoopModel(SectionModel):
    prng_sample: PRNGSample | None
    _target_class: ClassVar[Type] = PRNGLoop


_section_types = [
    SectionModel,
    MatchModel,
    AcquireLoopRtModel,
    SweepModel,
    CaseModel,
    PRNGSetupModel,
    PRNGLoopModel,
]
_section_types_target_class = {cl._target_class.__name__ for cl in _section_types}
AllSectionModel = Union[
    SectionModel,
    MatchModel,
    AcquireLoopRtModel,
    SweepModel,
    CaseModel,
    PRNGSetupModel,
    PRNGLoopModel,
]


def _unstructure_section_model(obj, _converter: Converter):
    if type(obj) is DoUntilSection:
        return _unstructure_do_until_section(obj, _converter)
    return unstructure_union_generic_type(obj, _section_types, _converter)


def _structure_section_model(d, _, _converter: Converter):
    if d.get("_type") == "DoUntilSection":
        return _structure_do_until_section(d, _converter)
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(d, _section_types, _converter)


def _unstructure_hqcs_condition(condition) -> dict | None:
    """Serialize the condition of a DoUntilSection or an `_Ensure`.

    - `_IsLive`: target (`Variable` or `Pulse`) cross-referenced by `$ref`.
    - Comparison predicates: not yet serializable; stored as `None`.
    """
    if isinstance(condition, _IsLive):
        return {
            "_kind": "is_live",
            "target": _unstructure_hqcs_target(condition.target),
        }
    return None


def _structure_hqcs_condition(d: dict | None):
    if d is None:
        return None
    return _IsLive(target=_structure_hqcs_target(d.get("target")))


def _unstructure_do_until_section(obj: "DoUntilSection", _converter: Converter) -> dict:
    """Serialize a DoUntilSection.

    Notes:
    - `condition`: an `is_live(...)` arrival predicate round-trips via
      `$ref` on its target; comparison predicates are not yet serializable
      and are stored as `None`.
    - `max_count`: stored faithfully.
    """
    children = [_unstructure_allsection_model_operation_model(c) for c in obj.children]

    return {
        "_type": "DoUntilSection",
        "uid": obj.uid,
        "name": obj.name,
        "alignment": _converter.unstructure(obj.alignment),
        "execution_type": obj.execution_type,
        "length": obj.length,
        "play_after": _unstructure_play_after_model(obj.play_after),
        "children": children,
        "trigger": obj.trigger,
        "section_timing_mode": _converter.unstructure(obj.section_timing_mode),
        "on_system_grid": obj.on_system_grid,
        "condition": _unstructure_hqcs_condition(obj.condition),
        "max_count": obj.max_count,
    }


def _structure_do_until_section(d: dict, _converter: Converter) -> "DoUntilSection":
    """Deserialize a DoUntilSection from a dict."""
    children = [
        _structure_allsection_model_operation_model(c, None)
        for c in d.get("children", [])
    ]

    section = DoUntilSection(
        uid=d.get("uid"),
        name=d.get("name", ""),
        alignment=_converter.structure(d.get("alignment"), SectionAlignmentModel),
        execution_type=d.get("execution_type"),
        length=d.get("length"),
        play_after=_structure_play_after_model(d.get("play_after"), None),
        children=children,
        trigger=d.get("trigger", {}),
        section_timing_mode=_converter.structure(
            d.get("section_timing_mode"), SectionTimingModeModel | None
        ),
        on_system_grid=d.get("on_system_grid"),
        condition=_structure_hqcs_condition(d.get("condition")),
        max_count=d.get("max_count", 1),
    )
    return section


# PRNG-related models


@attrs.define
class PRNGModel:
    range: int
    seed: int
    _target_class: ClassVar[Type] = PRNG


@attrs.define
class PRNGSampleModel:
    uid: str
    prng: PRNGModel
    count: int
    _target_class: ClassVar[Type] = PRNGSample


@attrs.define
class ExperimentSignalModel:
    uid: str
    calibration: SignalCalibrationModel | None
    mapped_logical_signal_path: str | None
    _target_class: ClassVar[Type] = ExperimentSignal


def _unstructure_allsection_model_operation_model(obj):
    if isinstance(obj, Section):
        return _converter.unstructure(obj, AllSectionModel)
    else:
        return _converter.unstructure(obj, OperationModel)


def _structure_allsection_model_operation_model(d, _):
    if d.get("_type") in _operation_types_target_class:
        return _converter.structure(d, OperationModel)
    else:
        return _converter.structure(d, AllSectionModel)


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
    _converter.register_unstructure_hook(
        OperationModel,
        partial(_unstructure_operation_model, _converter=_converter),
    )
    _converter.register_structure_hook(
        OperationModel,
        partial(_structure_operation_model, _converter=_converter),
    )

    _converter.register_unstructure_hook(PlayAfterModel, _unstructure_play_after_model)
    _converter.register_structure_hook(PlayAfterModel, _structure_play_after_model)

    _converter.register_unstructure_hook(MatchVariableModel, _unstructure_hqcs_target)
    _converter.register_structure_hook(
        MatchVariableModel, lambda d, _: _structure_hqcs_target(d)
    )

    _converter.register_unstructure_hook(
        AllSectionModel, partial(_unstructure_section_model, _converter=_converter)
    )

    _converter.register_structure_hook(
        AllSectionModel, partial(_structure_section_model, _converter=_converter)
    )

    _converter.register_unstructure_hook(
        AllSectionModel | OperationModel, _unstructure_allsection_model_operation_model
    )

    _converter.register_structure_hook(
        AllSectionModel | OperationModel, _structure_allsection_model_operation_model
    )

    _coprocessor.register(_converter)
    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
