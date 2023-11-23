# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import dataclasses
from typing import List, Mapping, Tuple, Union

from laboneq.data.path import Separator
from laboneq.data.setup_description import (
    DeviceType,
    Instrument,
    LogicalSignal,
    PhysicalChannel,
    Setup,
)
from laboneq.data.utils.calibration_helper import CalibrationHelper


def split_path(path: str) -> Tuple[str, str]:
    """Split path into group and name."""
    parts = path.split(Separator)
    return parts[-2], parts[-1]


class InstrumentHelper:
    def __init__(self, instruments: List[Instrument]):
        self._instruments = instruments
        self._ls_to_pc: Mapping[LogicalSignal, PhysicalChannel] = {}

    def physical_channel_by_logical_signal(
        self, logical_signal: Union[LogicalSignal, str]
    ) -> PhysicalChannel:
        if isinstance(logical_signal, str):
            *_, group, name = split_path(logical_signal)
            logical_signal = LogicalSignal(name=name, group=group)
        try:
            return self._ls_to_pc[logical_signal]
        except KeyError:
            for instr in self._instruments:
                for conn in instr.connections:
                    if conn.logical_signal == logical_signal:
                        self._ls_to_pc[logical_signal] = conn.physical_channel
                        return self._ls_to_pc[logical_signal]
        raise RuntimeError(
            f"Could not find physical channel for logical signal {logical_signal}"
        )


@dataclasses.dataclass
class PPCConnection:
    device: Instrument
    channel: int


class SetupHelper:
    def __init__(self, setup: Setup):
        # TODO: `Setup` is not hashable, thus cannot use `cached_method()` decorator
        self._setup = setup
        self._instrument_helper = InstrumentHelper(setup.instruments)
        self._calibration_helper = CalibrationHelper(setup.calibration)
        self._logical_signals: List[LogicalSignal] = []

    @property
    def instruments(self) -> InstrumentHelper:
        return self._instrument_helper

    @property
    def calibration(self) -> CalibrationHelper:
        return self._calibration_helper

    def logical_signals(self) -> List[LogicalSignal]:
        if self._logical_signals:
            return self._logical_signals
        for logical_signal_group in self._setup.logical_signal_groups.values():
            self._logical_signals.extend(
                [
                    logical_signal
                    for logical_signal in logical_signal_group.logical_signals.values()
                ]
            )
        return self._logical_signals

    def logical_signal_by_path(self, path: str) -> LogicalSignal:
        *_, group, name = split_path(path)
        return self._setup.logical_signal_groups[group].logical_signals[name]

    def ppc_connections(self) -> dict[LogicalSignal, PPCConnection]:
        ppc_conn_by_pc: dict[PhysicalChannel, PPCConnection] = {}
        ppc_conn_by_ls: dict[LogicalSignal, PPCConnection] = {}

        for conn in self._setup.setup_internal_connections:
            if (ppc := conn.from_instrument).device_type != DeviceType.SHFPPC:
                continue
            channel = conn.from_port.channel
            target_instrument = conn.to_instrument
            target_port = conn.to_port
            [physical_channel] = [
                pc
                for pc in target_instrument.physical_channels
                if target_port in pc.ports
            ]
            ppc_conn_by_pc[physical_channel] = PPCConnection(ppc, channel)

            ppc_conn_by_ls.update(
                {
                    ls_map.logical_signal: ppc_conn_by_pc[physical_channel]
                    for ls_map in target_instrument.connections
                    if ls_map.physical_channel == physical_channel
                }
            )

        return ppc_conn_by_ls
