# Copyright 2026 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Conversion of a DSL device setup into the QCCS capnp setup description."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from laboneq.core.exceptions import LabOneQException
from laboneq.core.path import remove_logical_signal_prefix
from laboneq.core.types.enums import IOSignalType
from laboneq.dsl.device.instruments import NonQC
from laboneq.dsl.device.instruments.zi_standard_instrument import ZIStandardInstrument

if TYPE_CHECKING:
    from laboneq.dsl.device.connection import Connection
    from laboneq.dsl.device.device_setup import DeviceSetup


#: Connection types that wire one instrument to another.
_INTERNAL_CONNECTION_TYPES = (IOSignalType.ZSYNC, IOSignalType.DIO, IOSignalType.PPC)


def build(device_setup: DeviceSetup) -> SimpleNamespace:
    """Builds the QCCS setup description."""
    instruments = [
        instrument
        for instrument in device_setup.instruments
        if isinstance(instrument, ZIStandardInstrument)
        and not isinstance(instrument, NonQC)
    ]
    signals = _signals(instruments)
    return SimpleNamespace(
        instruments=[_instrument(i) for i in instruments],
        signals=list(signals.values()),
        internal_connections=_pump_connections(instruments, signals),
    )


def _instrument(instrument: ZIStandardInstrument) -> SimpleNamespace:
    options = instrument.device_options
    clock = instrument.reference_clock_source
    return SimpleNamespace(
        uid=instrument.uid,
        device_type=instrument.calc_driver(),
        options=options.upper().split("/") if options else [],
        reference_clock_source=clock.name if clock else None,
    )


def _signals(instruments: list[ZIStandardInstrument]) -> dict[str, SimpleNamespace]:
    signals: dict[str, SimpleNamespace] = {}
    for instrument in instruments:
        for conn in instrument.connections:
            if conn.signal_type in _INTERNAL_CONNECTION_TYPES:
                continue
            uid = _signal_uid(conn)
            signal = signals.setdefault(
                uid, SimpleNamespace(uid=uid, ports=[], instrument_uid=instrument.uid)
            )
            paths = _port_paths(cast("str", conn.local_port))
            signal.ports.extend(path for path in paths if path not in signal.ports)
    return signals


def _pump_connections(
    instruments: list[ZIStandardInstrument], signals: dict[str, SimpleNamespace]
) -> list[SimpleNamespace]:
    connections: list[SimpleNamespace] = []
    for instrument in instruments:
        for conn in instrument.connections:
            if conn.signal_type != IOSignalType.PPC:
                continue
            uid = _signal_uid(conn)
            target = signals.get(uid)
            if target is None:
                raise LabOneQException(f"Logical signal '{uid}' not found.")
            connections.extend(
                SimpleNamespace(
                    from_instrument=instrument.uid,
                    from_port=conn.local_port,
                    to_instrument=target.instrument_uid,
                    to_port=port,
                )
                for port in target.ports
            )
    return connections


def _signal_uid(conn: Connection) -> str:
    return remove_logical_signal_prefix(cast("str", conn.remote_path))


def _port_paths(port: str) -> list[str]:
    # A UHFQA readout input spans both signal inputs.
    if port.startswith("QAS/"):
        return ["SIGINS/0", "SIGINS/1"]
    return [port]
