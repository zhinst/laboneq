# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from laboneq.core.types.enums.io_direction import IODirection
from laboneq.core.types.enums.io_signal_type import IOSignalType
from laboneq.data.execution_payload import (
    ServerType,
    TargetChannelCalibration,
    TargetChannelType,
    TargetDevice,
    TargetDeviceType,
    TargetServer,
    TargetSetup,
)

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.instruments.zi_standard_instrument import (
        ZIStandardInstrument,
    )


def convert_dsl_to_target_setup(device_setup: DeviceSetup) -> TargetSetup:
    servers: dict[str, TargetServer] = {
        server_uid: TargetServer(
            uid=server_uid,
            address=server.host,
            port=server.port,
            server_type=ServerType.DATA_SERVER,
            api_level=server.api_level,
        )
        for server_uid, server in device_setup.servers.items()
    }

    def _get_connected_outputs(instr: ZIStandardInstrument) -> dict[str, list[int]]:
        ls_to_ports: dict[str, list[int]] = {}
        for conn in instr.connections:
            if (
                conn.signal_type
                in (
                    IOSignalType.I,
                    IOSignalType.Q,
                    IOSignalType.IQ,
                    IOSignalType.RF,
                    IOSignalType.PPC,
                    IOSignalType.SINGLE,
                )
                and conn.direction == IODirection.OUT
            ):
                output_port = instr.output_by_uid(conn.local_port)
                dev_outputs = (
                    []
                    if output_port is None or output_port.physical_port_ids is None
                    else output_port.physical_port_ids
                )
                if dev_outputs:
                    ports = ls_to_ports.setdefault(conn.remote_path, [])
                    ports.extend([int(n) for n in dev_outputs])
        return ls_to_ports

    def _get_internal_connections(instr: ZIStandardInstrument) -> list[tuple[str, str]]:
        return [
            (c.local_port, c.remote_path)
            for c in instr.connections
            if c.signal_type in [IOSignalType.DIO, IOSignalType.ZSYNC]
        ]

    def _get_calibrations(
        instr: ZIStandardInstrument,
    ) -> list[TargetChannelCalibration]:
        target_calibs = []
        for conn in instr.connections:
            if conn.signal_type == IOSignalType.RF:
                voltage_offset = None
                port = None
                for lsg in device_setup.logical_signal_groups.values():
                    for ls in lsg.logical_signals.values():
                        if ls.path == conn.remote_path and ls.calibration:
                            voltage_offset = ls.calibration.voltage_offset
                            port = "/".join(ls.physical_channel.name.upper().split("_"))
                if voltage_offset is not None:
                    target_calibs.append(
                        TargetChannelCalibration(
                            channel_type=TargetChannelType.RF,
                            ports=[port],
                            voltage_offset=voltage_offset,
                        )
                    )
        return target_calibs

    def _instr_to_target_device(instr: ZIStandardInstrument) -> TargetDevice:
        options = instr.calc_options()
        return TargetDevice(
            uid=instr.uid,
            server=servers[instr.server_uid],
            device_serial=instr.address,
            device_type=next(
                (t for t in TargetDeviceType if t.name == instr.calc_driver()),
                TargetDeviceType.NONQC,
            ),
            interface=instr.interface,
            has_signals=bool(instr.connections),
            connected_outputs=_get_connected_outputs(instr),
            internal_connections=_get_internal_connections(instr),
            calibrations=_get_calibrations(instr),
            is_qc=options.get("is_qc", False),
            qc_with_qa=options.get("qc_with_qa", False),
            reference_clock_source=instr.reference_clock_source.name
            if instr.reference_clock_source
            else None,
        )

    target_setup = TargetSetup(
        uid=device_setup.uid,
        servers=list(servers.values()),
        devices=[_instr_to_target_device(instr) for instr in device_setup.instruments],
    )
    return target_setup
