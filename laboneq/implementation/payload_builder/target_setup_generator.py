# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Iterator

from laboneq.data.execution_payload import (
    TargetChannelCalibration,
    TargetChannelType,
    TargetDevice,
    TargetDeviceType,
    TargetServer,
    TargetSetup,
)
from laboneq.data.setup_description import (
    DeviceType,
    Instrument,
    IODirection,
    PhysicalChannelType,
    Server,
    Setup,
)
from laboneq.data.utils.calibration_helper import CalibrationHelper


class TargetSetupGenerator:
    @classmethod
    def from_setup(cls, setup: Setup, with_calibration: bool = True) -> TargetSetup:
        """Return a TargetSetup from the given device setup.

        Currently calibration is included in the TargetSetup but the intention is
        that it will be split out into its own data object and excluded from
        the execution payload in future. The `with_calibration` parameter is
        provided to make this change simpler.

        Args:
            setup:
                The device setup to generate a target setup for.
            with_calibration:
                Whether to include the calibration. If false, the `.calibration`
                attributes of the returned `TargetDevice` instances are set to `None`.

        Returns:
            target_setup: The generated target setup.
        """
        assert isinstance(setup, Setup)
        servers = [
            cls._target_server_from_server(server) for server in setup.servers.values()
        ]
        server_lookup = {server.uid: server for server in servers}
        devices = [
            d
            for instrument in setup.instruments
            for d in cls._target_devices_from_instrument(
                instrument,
                server_lookup[instrument.server.uid],
                setup,
                with_calibration=with_calibration,
            )
        ]
        cls._split_shfqc_zsync(devices, setup)

        return TargetSetup(
            uid=setup.uid,
            servers=servers,
            devices=devices,
        )

    @classmethod
    def _target_server_from_server(cls, server: Server) -> TargetServer:
        assert isinstance(server, Server)
        return TargetServer(
            uid=server.uid,
            host=server.host,
            port=server.port,
        )

    @classmethod
    def _convert_to_target_device_type(cls, dt: DeviceType) -> TargetDeviceType | None:
        if dt == DeviceType.UNMANAGED:
            return TargetDeviceType.NONQC
        try:
            return TargetDeviceType[dt.name]
        except KeyError:
            pass
        return None

    @staticmethod
    def split_shfqc(shfqc: Instrument):
        assert shfqc.device_type == DeviceType.SHFQC

        shfqa = Instrument(
            uid=f"{shfqc.uid}_shfqa",
            address=shfqc.address,
            device_type=DeviceType.SHFQA,
            server=shfqc.server,
            interface=shfqc.interface,
            reference_clock=shfqc.reference_clock,
        )
        shfsg = Instrument(
            uid=f"{shfqc.uid}_shfsg",
            address=shfqc.address,
            device_type=DeviceType.SHFSG,
            server=shfqc.server,
            interface=shfqc.interface,
            reference_clock=shfqc.reference_clock,
        )
        for port in shfqc.ports:
            if "SGCHANNELS" not in port.path:
                shfqa.ports.append(port)
            if "QACHANNELS" not in port.path:
                shfsg.ports.append(port)

        for device in shfsg, shfqa:
            device.physical_channels = [
                pc
                for pc in shfqc.physical_channels
                if any(port in device.ports for port in pc.ports)
            ]
            device.connections = [
                conn
                for conn in shfqc.connections
                if conn.physical_channel in device.physical_channels
            ]

        return shfsg, shfqa

    @classmethod
    def _target_devices_from_instrument(
        cls,
        instrument: Instrument,
        server: TargetServer,
        setup: Setup,
        with_calibration: bool,
    ) -> Iterator[TargetDevice]:
        if instrument.device_type == DeviceType.SHFQC:
            shfsg, shfqa = cls.split_shfqc(instrument)

            connected_outputs = cls._connected_outputs_from_instrument(shfqa)
            internal_connections = cls._internal_connections(shfqa, setup)
            if with_calibration:
                calibrations = cls._target_calibrations_from_instrument(shfqa, setup)
            else:
                calibrations = None

            target_device_qa = (
                TargetDevice(
                    uid=instrument.uid,
                    server=server,
                    device_serial=instrument.address,
                    device_type=TargetDeviceType.SHFQA,
                    device_options=instrument.device_options,
                    interface=instrument.interface,
                    has_signals=len(instrument.connections) > 0,
                    connected_outputs=connected_outputs,
                    internal_connections=internal_connections,
                    calibrations=calibrations,
                    is_qc=True,
                    qc_with_qa=True,
                    reference_clock_source=instrument.reference_clock.source,
                    device_class=instrument.device_class,
                )
                if shfqa.connections
                else None
            )
            if target_device_qa is not None:
                yield target_device_qa

            connected_outputs = cls._connected_outputs_from_instrument(shfsg)
            internal_connections = cls._internal_connections(shfsg, setup)
            if with_calibration:
                calibrations = cls._target_calibrations_from_instrument(shfsg, setup)
            else:
                calibrations = None

            uid = (
                f"{instrument.uid}_sg"
                if target_device_qa is not None
                else instrument.uid
            )

            target_device_sg = TargetDevice(
                uid=uid,
                server=server,
                device_serial=instrument.address,
                device_type=TargetDeviceType.SHFSG,
                device_options=instrument.device_options,
                interface=instrument.interface,
                has_signals=len(instrument.connections) > 0,
                connected_outputs=connected_outputs,
                internal_connections=internal_connections,
                calibrations=calibrations,
                is_qc=True,
                qc_with_qa=target_device_qa is not None,
                reference_clock_source=instrument.reference_clock.source,
                device_class=instrument.device_class,
            )
            yield target_device_sg
            return

        connected_outputs = cls._connected_outputs_from_instrument(instrument)
        internal_connections = cls._internal_connections(instrument, setup)
        if with_calibration:
            calibrations = cls._target_calibrations_from_instrument(instrument, setup)
        else:
            calibrations = None

        device_type = cls._convert_to_target_device_type(instrument.device_type)
        if device_type is not None:
            yield TargetDevice(
                uid=instrument.uid,
                server=server,
                # Here we translate from a theoretical instrument address
                # to a device serial number. For all ZI devices these
                # are the same:
                device_serial=instrument.address,
                device_type=device_type,
                device_options=instrument.device_options,
                interface=instrument.interface,
                has_signals=len(instrument.connections) > 0,
                connected_outputs=connected_outputs,
                internal_connections=internal_connections,
                calibrations=calibrations,
                is_qc=False,
                qc_with_qa=False,
                reference_clock_source=instrument.reference_clock.source,
                device_class=instrument.device_class,
            )

    @classmethod
    def _split_shfqc_zsync(cls, instruments: list[TargetDevice], setup: Setup):
        """After splitting any SHFQCs, also look for PQSC and updates its ZSync connection
        to also represent both parts."""

        split_qcs = {
            i.uid.removesuffix("_sg") for i in instruments if i.is_qc and i.qc_with_qa
        }

        for d in instruments:
            if d.device_type != TargetDeviceType.PQSC:
                continue

            extra_connections = []
            for path, instr_uid in d.internal_connections:
                if instr_uid in split_qcs:
                    extra_connections.append((path, f"{instr_uid}_sg"))

            d.internal_connections.extend(extra_connections)

    @classmethod
    def _connected_outputs_from_instrument(
        cls,
        instrument: Instrument,
        port_path_filter=("SIGOUTS", "SGCHANNELS", "QACHANNELS"),
    ) -> dict[str, list[int]]:
        ls_ports = {}
        for c in instrument.connections:
            if c.physical_channel.direction != IODirection.OUT:
                continue
            ports = [
                int(port.path.split("/")[1])
                for port in c.physical_channel.ports
                if any(port.path.startswith(s) for s in port_path_filter)
            ]
            if ports:
                ls_ports.setdefault(
                    f"/logical_signal_groups/{c.logical_signal.group}/{c.logical_signal.name}",
                    [],
                ).extend(ports)
        return ls_ports

    @classmethod
    def _internal_connections(
        cls, instrument: Instrument, setup: Setup
    ) -> list[tuple[str, str]]:
        return [
            (c.from_port.path, c.to_instrument.uid)
            for c in setup.setup_internal_connections
            if c.from_instrument.uid == instrument.uid
        ]

    @classmethod
    def _target_calibrations_from_instrument(
        cls,
        instrument: Instrument,
        setup: Setup,
    ) -> list[TargetChannelCalibration]:
        calibration = CalibrationHelper(setup.calibration)
        calibrations = []
        if calibration.empty():
            return calibrations
        for c in instrument.connections:
            sig_calib = calibration.by_logical_signal(c.logical_signal)
            if sig_calib is not None and sig_calib.voltage_offset is not None:
                channel_type = {
                    PhysicalChannelType.IQ_CHANNEL: TargetChannelType.IQ,
                    PhysicalChannelType.RF_CHANNEL: TargetChannelType.RF,
                }.get(c.physical_channel.type, TargetChannelType.UNKNOWN)
                calibrations.append(
                    TargetChannelCalibration(
                        channel_type=channel_type,
                        ports=[p.path for p in c.physical_channel.ports],
                        voltage_offset=sig_calib.voltage_offset,
                    )
                )
        return calibrations
