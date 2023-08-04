# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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
            cls._target_device_from_instrument(
                instrument,
                server_lookup[instrument.server.uid],
                setup,
                with_calibration=with_calibration,
            )
            for instrument in setup.instruments
        ]
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
            api_level=server.api_level,
        )

    @classmethod
    def _convert_to_target_device_type(cls, dt: DeviceType) -> TargetDeviceType:
        return TargetDeviceType[dt.name]

    @classmethod
    def _target_device_from_instrument(
        cls,
        instrument: Instrument,
        server: Server,
        setup: Setup,
        with_calibration: bool,
    ) -> TargetDevice:
        device_type = cls._convert_to_target_device_type(instrument.device_type)
        connected_outputs = cls._connected_outputs_from_instrument(instrument)
        internal_connections = cls._internal_connections(instrument, setup)
        if with_calibration:
            calibrations = cls._target_calibrations_from_instrument(instrument, setup)
        else:
            calibrations = None
        return TargetDevice(
            uid=instrument.uid,
            server=server,
            # Here we translate from a theoretical instrument address
            # to a device serial number. For all ZI devices these
            # are the same:
            device_serial=instrument.address,
            device_type=device_type,
            interface=instrument.interface,
            has_signals=len(instrument.connections) > 0,
            # ...
            connected_outputs=connected_outputs,
            internal_connections=internal_connections,
            calibrations=calibrations,
            is_qc=False,  # XXX: handle this here or in the device setup?
            qc_with_qa=False,  # XXX: handle this here or in the device setup?
            reference_clock_source=instrument.reference_clock.source,
        )

    @classmethod
    def _connected_outputs_from_instrument(
        cls, instrument: Instrument
    ) -> dict[str, list[int]]:
        ls_ports = {}
        for c in instrument.connections:
            ports = []
            for port in c.physical_channel.ports:
                if (
                    port.path.startswith("SIGOUTS")
                    or port.path.startswith("SGCHANNELS")
                    or port.path.startswith("QACHANNELS")
                ):
                    ports.append(int(port.path.split("/")[1]))
            if ports:
                ls_ports.setdefault(
                    f"{c.logical_signal.group}/{c.logical_signal.name}", []
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
            if sig_calib is not None:
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
