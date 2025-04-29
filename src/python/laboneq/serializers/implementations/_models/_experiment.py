# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from enum import Enum
from functools import partial
from typing import Callable, ClassVar, Optional, Type, Union

import attrs
from cattrs import Converter

from laboneq._utils import UIDReference
from laboneq.core.types.enums.acquisition_type import AcquisitionType
from laboneq.core.types.enums.averaging_mode import AveragingMode
from laboneq.core.types.enums.dsl_version import DSLVersion
from laboneq.core.types.enums.execution_type import ExecutionType
from laboneq.core.types.enums.repetition_mode import RepetitionMode
from laboneq.core.types.enums.section_alignment import SectionAlignment
from laboneq.dsl.experiment.acquire import Acquire
from laboneq.dsl.experiment.call import Call
from laboneq.dsl.experiment.delay import Delay
from laboneq.dsl.experiment.experiment_signal import ExperimentSignal
from laboneq.dsl.experiment.play_pulse import PlayPulse
from laboneq.dsl.experiment.pulse import PulseFunctional, PulseSampled
from laboneq.dsl.experiment.reserve import Reserve
from laboneq.dsl.experiment.section import (
    AcquireLoopNt,
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
from laboneq.serializers._cache import PulseSampledCache
from laboneq.serializers.implementations._models._calibration import (
    _structure_basic_or_parameter_model,
    _unstructure_basic_or_parameter_model,
)

from ._calibration import (
    ParameterModel,
    SignalCalibrationModel,
)
from ._calibration import (
    make_converter as make_converter_calibration,
)
from ._common import (
    ArrayLike_Model,
    NumericModel,
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
        # in which no reordering is done after converting the objects to dicts,
        # which is also a sequential process.
        _cache = PulseSampledCache.get_pulse_cache()
        if _cache.get(obj["$ref"]) is None:
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
    amplitude: Optional[NumericModel]
    length: float
    can_compress: bool
    pulse_parameters: Optional[PulseParameterModel]
    _target_class: ClassVar[Type] = PulseFunctional

    # simple enough to not require customized unstructure
    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            function=obj["function"],
            uid=obj["uid"],
            amplitude=_converter.structure(obj["amplitude"], Optional[NumericModel]),
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


# assume to be a dict with simple types
PulseParameterModel = dict[str, Union[float, int, str, bool, complex, ParameterModel]]


def _unstructure_pulse_parameter_model(obj):
    return {
        k: _unstructure_basic_or_parameter_model(v, _converter)
        if isinstance(v, (SweepParameter, LinearSweepParameter))
        else v
        for k, v in obj.items()
    }


def _structure_pulse_parameter_model(obj):
    ret = {}
    for k, v in obj.items():
        if isinstance(v, dict) and v.get("_type") is not None:
            ret[k] = _converter.structure(v, ParameterModel)
        else:
            ret[k] = v
    return ret


# Operation-related models
@attrs.define
class PlayPulseModel:
    signal: str
    pulse: PulseModel
    amplitude: Union[NumericModel, ParameterModel]
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
        amplitude = _structure_basic_or_parameter_model(obj["amplitude"], _converter)
        return cls._target_class(
            signal=obj["signal"],
            pulse=_converter.structure(obj["pulse"], PulseModel),
            amplitude=amplitude,
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


@attrs.define
class ReserveModel:
    signal: str
    _target_class: ClassVar[Type] = Reserve

    # simple enough to not require customized unstructure
    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(signal=obj["signal"])


@attrs.define
class SetNodeModel:
    path: str
    # Assume that the value is a simple type
    value: Optional[Union[str, int, float, complex, bool]]
    _target_class: ClassVar[Type] = SetNode

    # simple enough to not require customized unstructure

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(path=obj["path"], value=obj["value"])


@attrs.define
class DelayModel:
    signal: str | None
    time: float | ParameterModel | None
    precompensation_clear: bool | None
    _target_class: ClassVar[Type] = Delay

    @classmethod
    def _unstructure(cls, obj):
        return {
            "signal": obj.signal,
            "time": _unstructure_basic_or_parameter_model(obj.time, _converter),
            "precompensation_clear": obj.precompensation_clear,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            signal=obj["signal"],
            time=_structure_basic_or_parameter_model(obj["time"], _converter),
            precompensation_clear=obj["precompensation_clear"],
        )


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
class UIDReferenceModel:
    """Reference to an object with an UID.

    Args:
        uid: UID of the referenced object.
    """

    uid: str
    _target_class: ClassVar[Type] = UIDReference

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(uid=obj["uid"])


@attrs.define
class CallModel:
    func_name: str | Callable
    # args could be anything, here we limit it to simple types
    # and UIDReference
    args: dict[
        str,
        ParameterModel | str | int | float | bool | complex | None | UIDReferenceModel,
    ]
    _target_class: ClassVar[Type] = Call

    @classmethod
    def _unstructure(cls, obj):
        args = {
            k: _converter.unstructure(v, ParameterModel)
            if isinstance(v, (SweepParameter, LinearSweepParameter))
            else v
            for k, v in obj.args.items()
        }
        args = {}
        for k, v in obj.args.items():
            if isinstance(v, UIDReference):
                args[k] = _converter.unstructure(v, UIDReferenceModel)
            elif isinstance(v, (SweepParameter, LinearSweepParameter)):
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
            if hasattr(v, "_type"):
                args[k] = _converter.structure(v, ParameterModel)
            elif isinstance(v, dict) and v.get("uid") is not None:
                args[k] = _converter.structure(v, UIDReferenceModel)
            else:
                args[k] = v
        return cls._target_class(
            func_name=obj["func_name"],
            **args,
        )


_operation_types = [
    AcquireModel,
    CallModel,
    DelayModel,
    PlayPulseModel,
    ReserveModel,
    SetNodeModel,
]
_operation_types_target_class = {cl._target_class.__name__ for cl in _operation_types}
OperationModel = Union[
    AcquireModel,
    CallModel,
    DelayModel,
    PlayPulseModel,
    ReserveModel,
    SetNodeModel,
]

# When dropping support for Python 3.9, remove the following lines
# and use configure_tagged_union(OperationModel, _converter) instead


def _unstructure_operation_model(obj, _converter: Converter):
    return unstructure_union_generic_type(obj, _operation_types, _converter)


def _structure_operation_model(d, _, _converter: Converter):
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(d, _operation_types, _converter)


@attrs.define
class SectionModel:
    uid: str | None
    name: str
    alignment: SectionAlignment
    execution_type: ExecutionType | None
    length: float | None
    play_after: str | SectionModel | list[str | SectionModel] | None
    children: list[AllSectionModel | OperationModel]
    trigger: dict[str, dict[str, int]]

    on_system_grid: bool | None
    _target_class: ClassVar[Type] = Section

    @classmethod
    def _unstructure(cls, obj):
        if isinstance(obj.play_after, list):
            play_after = [p.uid for p in obj.play_after]
        elif obj.play_after is None or isinstance(obj.play_after, str):
            play_after = obj.play_after
        else:
            play_after = obj.play_after.uid

        children = []
        for c in obj.children:
            if isinstance(c, Section):
                children.append(_converter.unstructure(c, AllSectionModel))
            else:
                children.append(_converter.unstructure(c, OperationModel))

        return {
            "uid": obj.uid,
            "name": obj.name,
            "alignment": _converter.unstructure(obj.alignment, SectionAlignmentModel),
            "execution_type": _converter.unstructure(
                obj.execution_type, Optional[ExecutionTypeModel]
            ),
            "length": obj.length,
            "play_after": play_after,
            "children": children,
            "trigger": obj.trigger,
            "on_system_grid": obj.on_system_grid,
        }

    # TODO: Better name for this method
    # TODO: Use instead cattrs.strategies.include_subclasses() when
    # dropping support for Python 3.9
    @classmethod
    def _process_dict(cls, obj) -> dict:
        """Process the dictionary to make it suitable for the structure method.
        Will be used for structuring of subclasses of Section."""
        if isinstance(obj["play_after"], list):
            play_after = [
                p if isinstance(p, str) else _converter.structure(p, SectionModel)
                for p in obj["play_after"]
            ]
        elif obj["play_after"] is None or isinstance(obj["play_after"], str):
            play_after = obj["play_after"]
        else:
            play_after = (
                _converter.structure(obj["play_after"], SectionModel)
                if obj["play_after"]
                else None
            )

        # a bit hacky here to determine the type of the children
        # TODO: let cattrs automatically handle this when dropping support for 3.9
        children = []
        for c in obj["children"]:
            if c.get("_type") in _operation_types_target_class:
                children.append(_converter.structure(c, OperationModel))
            else:
                children.append(_converter.structure(c, AllSectionModel))

        return {
            "uid": obj["uid"],
            "name": obj["name"],
            "alignment": SectionAlignmentModel._target_class.value(obj["alignment"]),
            "execution_type": None
            if obj["execution_type"] is None
            else ExecutionTypeModel._target_class.value(obj["execution_type"]),
            "length": obj["length"],
            "play_after": play_after,
            "children": children,
            "trigger": obj["trigger"],
            "on_system_grid": obj["on_system_grid"],
        }

    @classmethod
    def _structure(cls, obj, _):
        d = cls._process_dict(obj)
        return cls._target_class(**d)


@attrs.define
class MatchModel:
    handle: str | None
    user_register: int | None
    prng_sample: PRNGSampleModel | None
    sweep_parameter: ParameterModel | None
    local: bool | None
    _target_class: ClassVar[Type] = Match

    @classmethod
    def _unstructure(cls, obj):
        ret = SectionModel._unstructure(obj)
        ret.update(
            {
                "handle": obj.handle,
                "user_register": obj.user_register,
                "prng_sample": _converter.unstructure(
                    obj.prng_sample, Optional[PRNGSampleModel]
                ),
                "sweep_parameter": _converter.unstructure(
                    obj.sweep_parameter,
                    ParameterModel,
                )
                if obj.sweep_parameter is not None
                else None,
                "local": obj.local,
            }
        )
        return ret

    @classmethod
    def _structure(cls, obj, _):
        ret = SectionModel._process_dict(obj)
        return cls._target_class(
            **ret,
            handle=obj["handle"],
            user_register=obj["user_register"],
            prng_sample=_converter.structure(
                obj["prng_sample"], Optional[PRNGSampleModel]
            ),
            sweep_parameter=_converter.structure(
                obj["sweep_parameter"],
                Optional[ParameterModel],
            )
            if obj["sweep_parameter"] is not None
            else None,
            local=obj["local"],
        )


@attrs.define
class AcquireLoopNtModel:
    averaging_mode: AveragingModeModel
    count: int
    execution_type: ExecutionTypeModel
    _target_class: ClassVar[Type] = AcquireLoopNt

    @classmethod
    def _unstructure(cls, obj):
        ret = SectionModel._unstructure(obj)
        ret.update(
            {
                "averaging_mode": obj.averaging_mode.value,
                "count": obj.count,
                "execution_type": obj.execution_type.value,
            }
        )
        return ret

    @classmethod
    def _structure(cls, obj, _):
        ret = SectionModel._process_dict(obj)
        ret.update(
            averaging_mode=AveragingModeModel._target_class.value(
                obj["averaging_mode"]
            ),
            count=obj["count"],
            execution_type=ExecutionTypeModel._target_class.value(
                obj["execution_type"]
            ),
        )
        return cls._target_class(**ret)


@attrs.define
class AcquireLoopRtModel:
    acquisition_type: AcquisitionTypeModel
    averaging_mode: AveragingModeModel
    count: int
    execution_type: ExecutionTypeModel
    repetition_mode: RepetitionModeModel
    repetition_time: float
    reset_oscillator_phase: bool
    _target_class: ClassVar[Type] = AcquireLoopRt

    @classmethod
    def _unstructure(cls, obj):
        ret = _converter.unstructure(obj, SectionModel)
        ret.update(
            {
                "acquisition_type": _converter.unstructure(
                    obj.acquisition_type, AcquisitionTypeModel
                ),
                "averaging_mode": _converter.unstructure(
                    obj.averaging_mode, AveragingModeModel
                ),
                "count": obj.count,
                "execution_type": _converter.unstructure(
                    obj.execution_type, ExecutionTypeModel
                ),
                "repetition_mode": _converter.unstructure(
                    obj.repetition_mode, RepetitionModeModel
                ),
                "repetition_time": obj.repetition_time,
                "reset_oscillator_phase": obj.reset_oscillator_phase,
            }
        )
        return ret

    @classmethod
    def _structure(cls, obj, _):
        ret = SectionModel._process_dict(obj)
        ret.update(
            acquisition_type=AcquisitionTypeModel._target_class.value(
                obj["acquisition_type"]
            ),
            averaging_mode=AveragingModeModel._target_class.value(
                obj["averaging_mode"]
            ),
            count=obj["count"],
            execution_type=ExecutionTypeModel._target_class.value(
                obj["execution_type"]
            ),
            repetition_mode=RepetitionModeModel._target_class.value(
                obj["repetition_mode"]
            ),
            repetition_time=obj["repetition_time"],
            reset_oscillator_phase=obj["reset_oscillator_phase"],
        )
        return cls._target_class(
            **ret,
        )


@attrs.define
class SweepModel:
    parameters: list[ParameterModel]
    reset_oscillator_phase: bool
    chunk_count: int
    _target_class: ClassVar[Type] = Sweep

    @classmethod
    def _unstructure(cls, obj):
        ret = _converter.unstructure(obj, SectionModel)
        ret.update(
            {
                "parameters": [
                    _converter.unstructure(p, ParameterModel) for p in obj.parameters
                ],
                "reset_oscillator_phase": obj.reset_oscillator_phase,
                "chunk_count": obj.chunk_count,
            }
        )
        return ret

    @classmethod
    def _structure(cls, obj, _):
        ret = SectionModel._process_dict(obj)
        ret.update(
            parameters=[
                _converter.structure(p, ParameterModel) for p in obj["parameters"]
            ],
            reset_oscillator_phase=obj["reset_oscillator_phase"],
            chunk_count=obj["chunk_count"],
        )
        return cls._target_class(**ret)


@attrs.define
class CaseModel:
    state: int
    _target_class: ClassVar[Type] = Case

    @classmethod
    def _unstructure(cls, obj):
        ret = _converter.unstructure(obj, SectionModel)
        ret.update({"state": obj.state})
        return ret

    @classmethod
    def _structure(cls, obj, _):
        d = SectionModel._process_dict(obj)
        d.update(state=obj["state"])
        return cls._target_class(**d)


@attrs.define
class PRNGSetupModel:
    prng: PRNGModel | None
    _target_class: ClassVar[Type] = PRNGSetup

    @classmethod
    def _unstructure(cls, obj):
        ret = _converter.unstructure(obj, SectionModel)
        ret.update({"prng": _converter.unstructure(obj.prng, Optional[PRNGModel])})
        return ret

    @classmethod
    def _structure(cls, obj, _):
        d = SectionModel._process_dict(obj)
        d.update(prng=_converter.structure(obj["prng"], Optional[PRNGModel]))
        return cls._target_class(**d)


@attrs.define
class PRNGLoopModel:
    prng_sample: PRNGSample | None
    _target_class: ClassVar[Type] = PRNGLoop

    @classmethod
    def _unstructure(cls, obj):
        ret = _converter.unstructure(obj, SectionModel)
        ret.update(
            {
                "prng_sample": _converter.unstructure(
                    obj.prng_sample, Optional[PRNGSample]
                )
            }
        )
        return ret

    @classmethod
    def _structure(cls, obj, _):
        d = SectionModel._process_dict(obj)
        d.update(
            prng_sample=_converter.structure(obj["prng_sample"], Optional[PRNGSample])
        )
        return cls._target_class(**d)


_section_types = [
    SectionModel,
    MatchModel,
    AcquireLoopNtModel,
    AcquireLoopRtModel,
    SweepModel,
    CaseModel,
    PRNGSetupModel,
    PRNGLoopModel,
]
AllSectionModel = Union[
    SectionModel,
    MatchModel,
    AcquireLoopNtModel,
    AcquireLoopRtModel,
    SweepModel,
    CaseModel,
    PRNGSetupModel,
    PRNGLoopModel,
]


def _unstructure_section_model(obj, _converter: Converter):
    return unstructure_union_generic_type(obj, _section_types, _converter)


def _structure_section_model(d, _, _converter: Converter):
    # cattrs requires the type to be passed as the second argument
    return structure_union_generic_type(d, _section_types, _converter)


# PRNG-related models


@attrs.define
class PRNGModel:
    range: int
    seed: int
    _target_class: ClassVar[Type] = PRNG

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(range=obj["range"], seed=obj["seed"])


@attrs.define
class PRNGSampleModel:
    uid: str
    prng: PRNGModel
    count: int
    _target_class: ClassVar[Type] = PRNGSample

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            prng=_converter.structure(obj["prng"], PRNGModel),
            count=obj["count"],
        )


@attrs.define
class ExperimentSignalModel:
    uid: str
    calibration: SignalCalibrationModel | None
    mapped_logical_signal_path: str | None
    _target_class: ClassVar[Type] = ExperimentSignal

    @classmethod
    def _unstructure(cls, obj):
        return {
            "uid": obj.uid,
            "calibration": _converter.unstructure(
                obj.calibration,
                Optional[SignalCalibrationModel],
            ),
            "mapped_logical_signal_path": obj.mapped_logical_signal_path,
        }

    @classmethod
    def _structure(cls, obj, _):
        return cls._target_class(
            uid=obj["uid"],
            calibration=_converter.structure(
                obj["calibration"], Optional[SignalCalibrationModel]
            ),
            mapped_logical_signal_path=obj["mapped_logical_signal_path"],
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
    _converter.register_unstructure_hook(
        OperationModel,
        partial(_unstructure_operation_model, _converter=_converter),
    )
    _converter.register_structure_hook(
        OperationModel,
        partial(_structure_operation_model, _converter=_converter),
    )

    _converter.register_unstructure_hook(
        AllSectionModel, partial(_unstructure_section_model, _converter=_converter)
    )

    _converter.register_structure_hook(
        AllSectionModel, partial(_structure_section_model, _converter=_converter)
    )

    register_models(_converter, collect_models(sys.modules[__name__]))
    return _converter


_converter = make_converter()
