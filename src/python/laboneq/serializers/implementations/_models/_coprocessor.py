# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from functools import partial
from typing import ClassVar, Type, Union

import attrs
import pybase64

from laboneq.dsl.coprocessor.stream import (
    InboundStream,
    OutboundStream,
    _FieldAccessor,
    _InboundPulseField,
    _InboundScalarField,
    _OutboundHandlesField,
    _OutboundLiteralField,
)
from laboneq.dsl.coprocessor.struct import Struct
from laboneq.dsl.variable import Variable
from laboneq.dsl.variable.types import (
    Amplitude,
    AmplitudeValue,
    DiscriminationDataPacked,
    Frequency,
    FrequencyValue,
    Int8,
    Int16,
    Int32,
    Int64,
    IqDataPacked,
    Phase,
    PhaseValue,
    ScopeShot,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    WaveformUpdate,
    _VarType,
)
from laboneq.serializers._cache import (
    StreamCache,
    VariableCache,
)

from ._common import (
    collect_models,
    register_models,
    structure_union_generic_type,
    unstructure_union_generic_type,
)


def _exp_converter():
    """The experiment converter, fetched lazily to break the import cycle.

    `_experiment` imports this module for its models, so we cannot import it
    at module load time. The converter is only needed at (un)structure time,
    by which point both modules are fully initialized. This mirrors how the
    models in `_experiment` reach their module-level `_converter`.
    """
    from laboneq.serializers.implementations._models._experiment import _converter

    return _converter


# ---------------------------------------------------------------------------
# Closed type catalog (name <-> class)
# ---------------------------------------------------------------------------

_ALL_TYPES: tuple[type[_VarType], ...] = (
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    DiscriminationDataPacked,
    IqDataPacked,
    ScopeShot,
    WaveformUpdate,
    Phase,
    Frequency,
    Amplitude,
)

_COPROC_TYPE_BY_NAME: dict[str, type[_VarType]] = {
    cls.__name__: cls for cls in _ALL_TYPES
}


def unstructure_coproc_type(cls: type[_VarType]) -> str:
    return cls.__name__


def structure_coproc_type(name: str) -> type[_VarType]:
    t = _COPROC_TYPE_BY_NAME.get(name)
    if t is None:
        raise ValueError(
            f"Unknown HQCS catalog type {name!r}. "
            f"Valid names: {sorted(_COPROC_TYPE_BY_NAME)}"
        )
    return t


# ---------------------------------------------------------------------------
# Variable.initial rich-value encoding
# ---------------------------------------------------------------------------


def _unstructure_initial(value):
    """Encode a Variable initial value (rich value object or basic scalar)."""
    if isinstance(value, PhaseValue):
        return {"_type": "PhaseValue", "radians": value.radians}
    if isinstance(value, FrequencyValue):
        return {"_type": "FrequencyValue", "hz": value.hz}
    if isinstance(value, AmplitudeValue):
        return {"_type": "AmplitudeValue", "value": value.value}
    # Basic scalars (int / float / None) pass through.
    return value


def _structure_initial(value):
    if isinstance(value, dict):
        kind = value["_type"]
        if kind == "PhaseValue":
            return PhaseValue(radians=value["radians"])
        if kind == "FrequencyValue":
            return FrequencyValue(hz=value["hz"])
        if kind == "AmplitudeValue":
            return AmplitudeValue(value=value["value"])
        raise ValueError(f"Unknown HQCS initial-value kind {kind!r}")
    return value


# ---------------------------------------------------------------------------
# Coprocessor payloads (opaque bytes, keyed by coprocessor label)
# ---------------------------------------------------------------------------


def unstructure_payloads(payloads: dict[str, bytes]) -> dict[str, str]:
    return {
        label: pybase64.b64encode(payload).decode("ascii")
        for label, payload in payloads.items()
    }


def structure_payloads(payloads: dict[str, str]) -> dict[str, bytes]:
    return {
        label: pybase64.b64decode(payload.encode("ascii"))
        for label, payload in payloads.items()
    }


# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------


@attrs.define
class StructModel:
    _target_class: ClassVar[Type] = Struct

    @classmethod
    def _unstructure(cls, obj):
        return {
            "fields": {n: unstructure_coproc_type(t) for n, t in obj.fields.items()}
        }

    @classmethod
    def _structure(cls, obj, _):
        return Struct(
            **{n: structure_coproc_type(tn) for n, tn in obj["fields"].items()}
        )


# ---------------------------------------------------------------------------
# Variable (cached for identity-preserving cross-references)
# ---------------------------------------------------------------------------


@VariableCache.cache
@attrs.define
class VariableModel:
    _target_class: ClassVar[Type] = Variable

    @classmethod
    def _unstructure(cls, obj):
        return {
            "type": unstructure_coproc_type(obj.type),
            "name": obj.name,
            "initial": _unstructure_initial(obj.initial),
            "log_handle": obj.log_handle,
        }

    @classmethod
    def _structure(cls, obj, _):
        var = Variable(
            type=structure_coproc_type(obj["type"]),
            name=obj.get("name"),
            initial=_structure_initial(obj.get("initial")),
        )
        var.log_handle = obj.get("log_handle")
        return var


# ---------------------------------------------------------------------------
# Field accessors
# ---------------------------------------------------------------------------


@attrs.define
class FieldAccessorModel:
    _target_class: ClassVar[Type] = _FieldAccessor

    @classmethod
    def _unstructure(cls, obj):
        return {"name": obj.name, "type": unstructure_coproc_type(obj.type)}

    @classmethod
    def _structure(cls, obj, _):
        return _FieldAccessor(name=obj["name"], type=structure_coproc_type(obj["type"]))


@attrs.define
class OutboundHandlesFieldModel:
    _target_class: ClassVar[Type] = _OutboundHandlesField

    @classmethod
    def _unstructure(cls, obj):
        return {
            "name": obj.name,
            "type": unstructure_coproc_type(obj.type),
            "handles": list(obj.handles),
        }

    @classmethod
    def _structure(cls, obj, _):
        acc = _OutboundHandlesField(
            name=obj["name"], type=structure_coproc_type(obj["type"])
        )
        acc.handles = list(obj.get("handles", []))
        return acc


@attrs.define
class OutboundLiteralFieldModel:
    _target_class: ClassVar[Type] = _OutboundLiteralField

    @classmethod
    def _unstructure(cls, obj):
        return {"name": obj.name, "type": unstructure_coproc_type(obj.type)}

    @classmethod
    def _structure(cls, obj, _):
        return _OutboundLiteralField(
            name=obj["name"], type=structure_coproc_type(obj["type"])
        )


@attrs.define
class InboundScalarFieldModel:
    _target_class: ClassVar[Type] = _InboundScalarField

    @classmethod
    def _unstructure(cls, obj):
        target = (
            _exp_converter().unstructure(obj.target, VariableModel)
            if obj.target is not None
            else None
        )
        return {
            "name": obj.name,
            "type": unstructure_coproc_type(obj.type),
            "target": target,
        }

    @classmethod
    def _structure(cls, obj, _):
        acc = _InboundScalarField(
            name=obj["name"], type=structure_coproc_type(obj["type"])
        )
        if obj.get("target") is not None:
            acc.target = _exp_converter().structure(obj["target"], VariableModel)
        return acc


@attrs.define
class InboundPulseFieldModel:
    _target_class: ClassVar[Type] = _InboundPulseField

    @classmethod
    def _unstructure(cls, obj):
        # Lazy import: PulseModel lives in _experiment (see _exp_converter).
        from ._experiment import PulseModel

        target = (
            _exp_converter().unstructure(obj.target, PulseModel)
            if obj.target is not None
            else None
        )
        return {
            "name": obj.name,
            "type": unstructure_coproc_type(obj.type),
            "target": target,
        }

    @classmethod
    def _structure(cls, obj, _):
        from ._experiment import PulseModel

        acc = _InboundPulseField(
            name=obj["name"], type=structure_coproc_type(obj["type"])
        )
        if obj.get("target") is not None:
            acc.target = _exp_converter().structure(obj["target"], PulseModel)
        return acc


_accessor_models = [
    OutboundHandlesFieldModel,
    OutboundLiteralFieldModel,
    InboundScalarFieldModel,
    InboundPulseFieldModel,
    FieldAccessorModel,
]
FieldAccessorModelUnion = Union[
    OutboundHandlesFieldModel,
    OutboundLiteralFieldModel,
    InboundScalarFieldModel,
    InboundPulseFieldModel,
    FieldAccessorModel,
]


def _unstructure_accessor(obj, converter):
    return unstructure_union_generic_type(obj, _accessor_models, converter)


def _structure_accessor(d, _, converter):
    return structure_union_generic_type(d, _accessor_models, converter)


# ---------------------------------------------------------------------------
# Streams (cached)
# ---------------------------------------------------------------------------


def _unstructure_stream(obj):
    conv = _exp_converter()
    return {
        "uid": obj.uid,
        "schema": conv.unstructure(obj.schema, StructModel),
        "src": obj.src,
        "dst": obj.dst,
        "link": obj.link,
        "fields": {
            n: conv.unstructure(a, FieldAccessorModelUnion)
            for n, a in obj.fields.items()
        },
    }


def _structure_stream(obj, target_class):
    conv = _exp_converter()
    # Bypass _Stream.__init__ — it rebuilds fields from the schema and (before
    # this refactor) assigned a per-instance id. Deserialization restores the
    # saved fields directly.
    stream = object.__new__(target_class)
    stream.uid = obj.get("uid")
    stream.schema = conv.structure(obj["schema"], StructModel)
    stream.src = obj.get("src")
    stream.dst = obj.get("dst")
    stream.link = obj.get("link")
    stream.fields = {
        n: conv.structure(fd, FieldAccessorModelUnion)
        for n, fd in obj.get("fields", {}).items()
    }
    return stream


@StreamCache.cache
@attrs.define
class OutboundStreamModel:
    _target_class: ClassVar[Type] = OutboundStream

    @classmethod
    def _unstructure(cls, obj):
        return _unstructure_stream(obj)

    @classmethod
    def _structure(cls, obj, _):
        return _structure_stream(obj, cls._target_class)


@StreamCache.cache
@attrs.define
class InboundStreamModel:
    _target_class: ClassVar[Type] = InboundStream

    @classmethod
    def _unstructure(cls, obj):
        return _unstructure_stream(obj)

    @classmethod
    def _structure(cls, obj, _):
        return _structure_stream(obj, cls._target_class)


_stream_models = [
    OutboundStreamModel,
    InboundStreamModel,
]
StreamModel = Union[
    OutboundStreamModel,
    InboundStreamModel,
]


def _unstructure_stream_model(obj, converter):
    return unstructure_union_generic_type(obj, _stream_models, converter)


def _structure_stream_model(d, _, converter):
    return structure_union_generic_type(d, _stream_models, converter)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register(converter) -> None:
    """Register all HQCS models and union hooks onto `converter`."""
    converter.register_unstructure_hook(
        StreamModel, partial(_unstructure_stream_model, converter=converter)
    )
    converter.register_structure_hook(
        StreamModel, partial(_structure_stream_model, converter=converter)
    )
    converter.register_unstructure_hook(
        FieldAccessorModelUnion, partial(_unstructure_accessor, converter=converter)
    )
    converter.register_structure_hook(
        FieldAccessorModelUnion, partial(_structure_accessor, converter=converter)
    )
    register_models(converter, collect_models(sys.modules[__name__]))
