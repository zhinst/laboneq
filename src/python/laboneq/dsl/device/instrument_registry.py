# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0


from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Iterator

import attrs

from laboneq.dsl.device import Instrument
from laboneq.dsl.device.connection import InternalConnection, SignalConnection

if TYPE_CHECKING:
    from laboneq.dsl.device.connection_builder import ConnectionBuilder

#: Builds a connection from one decoded entry of a descriptor's `connections`:
#: the keyword the entry uses, the target it names and the local ports it lists.
DescriptorConnectionDecoder = Callable[
    [str, str, list[str]], InternalConnection | SignalConnection
]

#: Builds the instrument for a device type no registered instrument claims, from
#: that device type and the fields the setup description gives for it.
UnmanagedInstrumentFactory = Callable[..., Instrument]


@attrs.define(frozen=True)
class InstrumentModel:
    """An instrument class, and how the framework handles it."""

    cls: type[Instrument]

    #: The key this instrument appears under in a descriptor, which is also the
    #: device type an instrument topology reports for it. `None` for an
    #: instrument that a descriptor cannot name.
    descriptor_key: str | None = None

    #: Builds this instrument's connections from the decoded entries of a
    #: descriptor. `None` to use the standard decoding of the descriptor's
    #: connection keywords.
    descriptor_connection_decoder: DescriptorConnectionDecoder | None = None

    #: Makes the connections a device setup declares for this instrument.
    #: `None` for an instrument that makes no connections.
    connection_builder: ConnectionBuilder | None = None


_registered_instruments: dict[str, InstrumentModel] = {}
_registered_unmanaged_instrument: UnmanagedInstrumentFactory | None = None


def register_instrument(
    cls: type[Instrument],
    *,
    descriptor_key: str | None = None,
    descriptor_connection_decoder: DescriptorConnectionDecoder | None = None,
    connection_builder: ConnectionBuilder | None = None,
) -> None:
    """Register an instrument class.

    Args:
        cls: The instrument class to register.
        descriptor_key: The key this instrument appears under in a descriptor,
            which is also the device type an instrument topology reports for it.
            `None` for an instrument that a descriptor cannot name.
        descriptor_connection_decoder: Builds this instrument's connections
            from the decoded entries of a descriptor. `None` to use the
            standard decoding of the descriptor's connection keywords.
        connection_builder: Makes the connections a device setup declares for
            this instrument. `None` for an instrument that makes no connections.
    """
    if not issubclass(cls, Instrument):
        raise TypeError(
            f"Instrument class must be a subclass of Instrument, got {cls}."
        )
    if cls.__name__ in _registered_instruments:
        raise ValueError(f"Instrument class {cls.__name__} is already registered.")
    _registered_instruments[cls.__name__] = InstrumentModel(
        cls=cls,
        descriptor_key=descriptor_key,
        descriptor_connection_decoder=descriptor_connection_decoder,
        connection_builder=connection_builder,
    )


def register_unmanaged_instrument(factory: UnmanagedInstrumentFactory) -> None:
    """Register the instrument that device types no other instrument claims
    fall back to.

    Args:
        factory: Builds the instrument from that device type and the fields the
            setup description gives for it.
    """
    global _registered_unmanaged_instrument
    if _registered_unmanaged_instrument is not None:
        raise ValueError("An unmanaged instrument is already registered.")
    _registered_unmanaged_instrument = factory


def unmanaged_instrument_factory() -> UnmanagedInstrumentFactory | None:
    """The instrument device types no registered instrument claims fall back to."""
    return _registered_unmanaged_instrument


def connection_builder(instrument: Instrument) -> ConnectionBuilder | None:
    """Makes the connections a device setup declares for this instrument.

    `None` if the instrument makes no connections, or if no backend registered it.
    """
    model = _registered_instruments.get(type(instrument).__name__)
    if model is None or model.cls is not type(instrument):
        return None
    return model.connection_builder


def model_by_descriptor_key(descriptor_key: str) -> InstrumentModel | None:
    """The instrument a descriptor names with this key, if any is registered."""
    return next(
        (
            model
            for model in _registered_instruments.values()
            if model.descriptor_key == descriptor_key
        ),
        None,
    )


def models_with_descriptor_key() -> Iterator[tuple[str, InstrumentModel]]:
    """Every instrument a descriptor can name, in registration order."""
    for model in _registered_instruments.values():
        if model.descriptor_key is not None:
            yield model.descriptor_key, model
