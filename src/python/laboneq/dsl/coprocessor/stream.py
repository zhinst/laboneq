# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

import attrs

from laboneq.dsl.variable import Variable
from laboneq.dsl.variable.types import (
    Amplitude,
    Direction,
    DiscriminationDataPacked,
    Frequency,
    Int8,
    Int16,
    Int32,
    Int64,
    IqDataPacked,
    Phase,
    ScopeShot,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    WaveformUpdate,
)

if TYPE_CHECKING:
    from laboneq.dsl.coprocessor.struct import Struct


@attrs.define
class _FieldAccessor:
    name: str
    type: type  # type[_VarType] but kept loose for attrs

    # We define methods, so the autocompletion sees them even when the type checker
    # is unable to resolve the exact type of the field accessor.

    def set_handles(self, handles: Sequence[str]):
        raise NotImplementedError(
            "`set_handles()` is only implemented for outbound measurement fields."
        )

    def as_variable(
        self,
        *,
        initial: Any | None = None,
        name: str | None = None,
        log_handle: str | None = None,
    ) -> Variable:
        raise NotImplementedError(
            "`as_variable()` is only implemented for inbound fields."
        )

    def updates(self, target: Variable):
        raise NotImplementedError("`updates()` is only implemented for inbound fields.")


@attrs.define
class _OutboundHandlesField(_FieldAccessor):
    handles: list[str] = attrs.field(factory=list)

    def set_handles(self, handles: Sequence[str]) -> None:
        """Bind this field to a list of acquisition handles.

        Each matching `acquire(handle=...)` feeds the value to the
        packetizer automatically; `send(...)` commits the accumulated values
        as one logical packet.
        """
        self.handles = list(handles)


@attrs.define
class _OutboundLiteralField(_FieldAccessor):
    """Outbound raw integer; value is supplied at `send` time as a kwarg."""


@attrs.define
class _InboundScalarField(_FieldAccessor):
    target: Variable | None = attrs.field(default=None, init=False)

    def updates(self, target: Variable) -> None:
        """Bind this field to a Variable. Inbound packets update its value."""
        self.target = target

    def as_variable(
        self,
        *,
        initial: Any | None = None,
        name: str | None = None,
        log_handle: str | None = None,
    ) -> Variable:
        """Create a fresh Variable of this field's type, bind, optionally log."""
        var = Variable(type=self.type, name=name, initial=initial)
        self.target = var
        if log_handle is not None:
            var.log(handle=log_handle)
        return var


@attrs.define
class _InboundPulseField(_FieldAccessor):
    target: Any | None = attrs.field(default=None, init=False)  # Pulse

    def updates(self, target: Any) -> None:
        """Bind this field to a Pulse. Inbound packets rewrite its samples."""
        self.target = target


_OUTBOUND_ACQUISITION_TYPES: set[type] = {
    DiscriminationDataPacked,
    IqDataPacked,
    ScopeShot,
}
_OUTBOUND_LITERAL_TYPES: set[type] = {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
}
_INBOUND_SCALAR_TYPES: set[type] = {
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Phase,
    Frequency,
    Amplitude,
}
_INBOUND_PULSE_TYPES: set[type] = {WaveformUpdate}


def _make_accessor(
    name: str,
    type_: type,
    direction: Direction,
) -> _FieldAccessor:
    """Build the right accessor object for a (type, direction) pair."""
    if direction is Direction.OUT:
        if type_ in _OUTBOUND_ACQUISITION_TYPES:
            return _OutboundHandlesField(name=name, type=type_)
        if type_ in _OUTBOUND_LITERAL_TYPES:
            return _OutboundLiteralField(name=name, type=type_)
    elif direction is Direction.IN:
        if type_ in _INBOUND_PULSE_TYPES:
            return _InboundPulseField(name=name, type=type_)
        if type_ in _INBOUND_SCALAR_TYPES:
            return _InboundScalarField(name=name, type=type_)
    return _FieldAccessor(name=name, type=type_)


class _Stream:
    """Base stream IR node. Subclassed per direction."""

    def __init__(
        self,
        *,
        uid: str | None = None,
        schema: Struct,
        src: str | None,
        dst: str | None,
        link: str | None,
        direction: Direction,
    ) -> None:
        self.uid = uid
        self.schema = schema
        self.src = src
        self.dst = dst
        self.link = link
        self.fields: dict[str, _FieldAccessor] = {
            name: _make_accessor(name, t, direction)
            for name, t in schema.fields.items()
        }

    def __eq__(self, other: object) -> bool:
        # Concrete-type match keeps OutboundStream != InboundStream.
        if self is other:
            return True
        if type(self) is not type(other):
            return NotImplemented
        return (
            self.uid == other.uid
            and self.schema == other.schema
            and self.src == other.src
            and self.dst == other.dst
            and self.link == other.link
            and self.fields == other.fields
        )


class OutboundStream(_Stream):
    """Stream from the LabOne Q control system to a coprocessor."""

    def __init__(
        self,
        *,
        schema: Struct,
        dst: str,
        link: str | None,
        uid: str | None = None,
    ) -> None:
        super().__init__(
            uid=uid,
            schema=schema,
            src=None,
            dst=dst,
            link=link,
            direction=Direction.OUT,
        )


class InboundStream(_Stream):
    """Stream from a coprocessor to the LabOne Q control system."""

    def __init__(
        self,
        *,
        schema: Struct,
        src: str,
        link: str | None,
        uid: str | None = None,
    ) -> None:
        super().__init__(
            uid=uid,
            schema=schema,
            src=src,
            dst=None,
            link=link,
            direction=Direction.IN,
        )
