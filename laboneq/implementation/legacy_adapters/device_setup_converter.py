# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Mapping, Tuple

from laboneq.core.exceptions import LabOneQException
from laboneq.core.types import enums as legacy_enums
from laboneq.data.setup_description import (
    ChannelMapEntry,
    DeviceType,
    Instrument,
    IODirection,
    LogicalSignal,
    LogicalSignalGroup,
    PhysicalChannel,
    PhysicalChannelType,
    Port,
    PortType,
    ReferenceClock,
    ReferenceClockSource,
    Server,
    Setup,
    SetupInternalConnection,
)
from laboneq.dsl import device as legacy_device
from laboneq.dsl.device import instruments as legacy_instruments
from laboneq.dsl.device import io_units as legacy_io_units
from laboneq.implementation.legacy_adapters import calibration_converter, utils
from laboneq.implementation.legacy_adapters.utils import (
    LogicalSignalPhysicalChannelUID,
    raise_not_implemented,
)
from laboneq.implementation.utils import devices

if TYPE_CHECKING:
    from laboneq.dsl.device import logical_signal_group as legacy_lsg
    from laboneq.dsl.device import servers as legacy_servers
    from laboneq.dsl.device.instruments.zi_standard_instrument import (
        ZIStandardInstrument,
    )


def convert_io_direction(obj: legacy_enums.io_direction.IODirection) -> IODirection:
    return IODirection[obj.name]


def convert_logical_signal(target: legacy_lsg.LogicalSignal) -> LogicalSignal:
    path = LogicalSignalPhysicalChannelUID(target.uid)
    return LogicalSignal(name=path.name, group=path.group)


def convert_logical_signal_groups_with_ls_mapping(
    logical_signal_groups: Dict[str, legacy_lsg.LogicalSignalGroup]
) -> Tuple[
    Dict[str, LogicalSignalGroup], Dict[legacy_lsg.LogicalSignal, LogicalSignal]
]:
    mapping = {}
    legacy_to_new = {}
    for uid, lsg in logical_signal_groups.items():
        new_lsg = LogicalSignalGroup(uid)
        for ls in lsg.logical_signals.values():
            new_ls = convert_logical_signal(ls)
            legacy_to_new[ls] = new_ls
            new_lsg.logical_signals[new_ls.name] = new_ls
        mapping[new_lsg.uid] = new_lsg
    return mapping, legacy_to_new


def convert_dataserver(target: legacy_servers.DataServer, leader_uid: str) -> Server:
    return Server(
        uid=target.uid,
        host=target.host,
        leader_uid=leader_uid,
        port=int(target.port),
    )


def convert_reference_clock_source(
    target: legacy_enums.ReferenceClockSource | None,
) -> ReferenceClockSource | None:
    if target is None:
        return None
    if target == legacy_enums.ReferenceClockSource.EXTERNAL:
        return ReferenceClockSource.EXTERNAL
    if target == legacy_enums.ReferenceClockSource.INTERNAL:
        return ReferenceClockSource.INTERNAL
    raise_not_implemented(target)


def convert_device_type(target: ZIStandardInstrument) -> DeviceType:
    if isinstance(target, legacy_instruments.HDAWG):
        return DeviceType.HDAWG
    if isinstance(target, legacy_instruments.SHFQA):
        return DeviceType.SHFQA
    if isinstance(target, legacy_instruments.SHFQC):
        return DeviceType.SHFQC
    if isinstance(target, legacy_instruments.PQSC):
        return DeviceType.PQSC
    if isinstance(target, legacy_instruments.UHFQA):
        return DeviceType.UHFQA
    if isinstance(target, legacy_instruments.SHFPPC):
        return DeviceType.SHFPPC
    if isinstance(target, legacy_instruments.SHFSG):
        return DeviceType.SHFSG
    if isinstance(target, legacy_instruments.PRETTYPRINTERDEVICE):
        return DeviceType.PRETTYPRINTERDEVICE
    if isinstance(target, legacy_instruments.NonQC):
        return DeviceType.UNMANAGED
    raise_not_implemented(target)


def convert_physical_channel_type(
    target: legacy_io_units.PhysicalChannelType | None,
) -> PhysicalChannelType | None:
    if target == legacy_io_units.PhysicalChannelType.IQ_CHANNEL:
        return PhysicalChannelType.IQ_CHANNEL
    if target == legacy_io_units.PhysicalChannelType.RF_CHANNEL:
        return PhysicalChannelType.RF_CHANNEL
    raise_not_implemented(target)


def convert_physical_channel(
    target: legacy_io_units.PhysicalChannel,
    direction: legacy_enums.io_direction.IODirection,
) -> PhysicalChannel:
    pc_helper = utils.LogicalSignalPhysicalChannelUID(target.uid)

    return PhysicalChannel(
        name=pc_helper.name,
        group=pc_helper.group,
        type=convert_physical_channel_type(target.type),
        direction=direction,
    )


class PortConverter:
    def __init__(self, target: ZIStandardInstrument):
        self._legacy_port_lookup = {port.uid: port for port in target.ports}
        self._ports = self.convert_instrument_ports(target)
        self._lookup = {port.path: port for port in self._ports}

    @property
    def ports(self) -> List[Port]:
        return self._ports

    def legacy_port_by_uid(self, uid: str) -> legacy_device.Port:
        return self._legacy_port_lookup[uid]

    @staticmethod
    def convert_instrument_ports(target: ZIStandardInstrument) -> List[Port]:
        if isinstance(target, legacy_instruments.HDAWG):
            return devices.hdawg_ports()
        elif isinstance(target, legacy_instruments.UHFQA):
            return devices.uhfqa_ports()
        elif isinstance(target, legacy_instruments.SHFPPC):
            return devices.shfppc_ports()
        elif isinstance(target, legacy_instruments.SHFQA):
            return devices.shfqa_ports()
        elif isinstance(target, legacy_instruments.SHFQC):
            return devices.shfqc_ports()
        elif isinstance(target, legacy_instruments.SHFSG):
            return devices.shfsg_ports()
        elif isinstance(target, legacy_instruments.PQSC):
            return devices.pqsc_ports()
        elif isinstance(target, legacy_instruments.PRETTYPRINTERDEVICE):
            return devices.test_device_ports()
        elif isinstance(target, legacy_instruments.NonQC):
            return devices.nonqc_ports()
        else:
            raise_not_implemented(target)

    def ports_by_legacy_uid(self, uid: str) -> List[Port]:
        port = self.legacy_port_by_uid(uid)
        # The connection is of type "IQ", so it refers to 2 ports
        if port.signal_type == legacy_enums.IOSignalType.IQ:
            if "QAS" in port.uid:
                uid = "SIGINS"
                return [self._lookup[uid + "/0"], self._lookup[uid + "/1"]]
        return [
            self._lookup[uid],
        ]


class LegacyConnectionFinder:
    def __init__(
        self,
        legacy_device: ZIStandardInstrument,
        legacy_ls: List[legacy_lsg.LogicalSignal],
    ):
        self.legacy_device = legacy_device
        self.connections = legacy_device.connections
        self._legacy_ls_lookup = {ls.path: ls for ls in legacy_ls}
        self._ports = {port.uid: port for port in legacy_device.ports}

    def connections_from_port_to_local_logical_signal(
        self,
    ) -> Tuple[legacy_lsg.LogicalSignal, legacy_device.Port]:
        """Logical signal connections that point to the device."""
        for conn in self.connections:
            if logical_signal := self._legacy_ls_lookup.get(conn.remote_path):
                # Port and LogicalSignal point to the same direction, should be the
                # same device.
                if port := self._ports[conn.local_port]:
                    if port.direction == logical_signal.direction:
                        yield logical_signal, port

    def connections_from_port_to_remote_logical_signal(
        self,
    ) -> Tuple[legacy_lsg.LogicalSignal, legacy_device.Port]:
        """Connections that point to the remote logical signal (not into this device)."""
        for conn in self.connections:
            remote_logical_signal = self._legacy_ls_lookup.get(conn.remote_path)
            # Port is defined to go to logical signal.
            if remote_logical_signal:
                from_port = self._ports[conn.local_port]
                if remote_logical_signal.direction != conn.direction:
                    yield remote_logical_signal, from_port
                else:
                    uid = utils.LogicalSignalPhysicalChannelUID(
                        remote_logical_signal.physical_channel.uid
                    )
                    if uid.group != self.legacy_device.uid:
                        raise LabOneQException(
                            "Instrument to instrument connection ports point to the"
                            f" same direction: {self.legacy_device.uid}/{from_port.uid}"
                            f" -> {remote_logical_signal.uid}"
                            f" -> {remote_logical_signal.physical_channel.uid}"
                        )

    def connections_from_port_to_another_device(self) -> Tuple[str, legacy_device.Port]:
        """Connections that go into the another device port."""
        for conn in self.connections:
            remote_logical_signal = self._legacy_ls_lookup.get(conn.remote_path)
            if not remote_logical_signal:
                from_port = self._ports[conn.local_port]
                if conn.signal_type in {
                    legacy_enums.IOSignalType.ZSYNC,
                    legacy_enums.IOSignalType.DIO,
                }:
                    yield conn.remote_path, from_port
                else:
                    raise NotImplementedError(
                        f"No known input port type in device {conn.remote_path} for"
                        f" port type {conn.signal_type}."
                    )


class InstrumentConverter:
    def __init__(
        self,
        src: ZIStandardInstrument,
        legacy_ls_to_new_map: Mapping[legacy_lsg.LogicalSignal, LogicalSignal],
        server: Server,
    ) -> None:
        self._src = src
        self._legacy_ls_to_new_map = legacy_ls_to_new_map
        self.port_converter = PortConverter(src)
        self.connection_converter = LegacyConnectionFinder(
            self._src, legacy_ls_to_new_map.keys()
        )
        self._server = server

    @cached_property
    def uid(self) -> str:
        return self._src.uid

    @cached_property
    def address(self) -> str:
        return self._src.address

    @cached_property
    def interface(self) -> str:
        return self._src.interface

    @cached_property
    def reference_clock(self) -> ReferenceClock:
        ref_clk = ReferenceClock()
        if self._src.reference_clock_source is not None:
            ref_clk.source = convert_reference_clock_source(
                self._src.reference_clock_source
            )
        if isinstance(self._src, legacy_instruments.PQSC):
            ref_clk.frequency = self._src.reference_clock
        return ref_clk

    @cached_property
    def ports(self) -> List[Port]:
        return self.port_converter.ports

    @cached_property
    def physical_channels(self) -> List[PhysicalChannel]:
        return list(self._physical_channels_map.values())

    @cached_property
    def _physical_channels_map(self) -> Mapping[str, PhysicalChannel]:
        physical_chs = {}
        for (
            logical_signal,
            port,
        ) in self.connection_converter.connections_from_port_to_local_logical_signal():
            phys_ch_uid = logical_signal.physical_channel.uid
            if phys_ch_uid not in physical_chs:
                direction = convert_io_direction(logical_signal.direction)
                physical_ch = convert_physical_channel(
                    logical_signal.physical_channel, direction
                )
                physical_chs[phys_ch_uid] = physical_ch
            for new_port in self.port_converter.ports_by_legacy_uid(port.uid):
                if new_port not in physical_chs[phys_ch_uid].ports:
                    physical_chs[phys_ch_uid].ports.append(new_port)
        return physical_chs

    @cached_property
    def channel_map_entries(self) -> List[ChannelMapEntry]:
        channel_map_entries = []
        for (
            logical_signal,
            _,
        ) in self.connection_converter.connections_from_port_to_local_logical_signal():
            phys_ch_uid = logical_signal.physical_channel.uid
            entry = ChannelMapEntry(
                physical_channel=self._physical_channels_map[phys_ch_uid],
                logical_signal=self._legacy_ls_to_new_map[logical_signal],
            )
            if entry not in channel_map_entries:
                channel_map_entries.append(entry)
        return channel_map_entries

    @property
    def instrument(self) -> Instrument:
        obj = Instrument(
            uid=self.uid,
            address=self.address,
            interface=self.interface,
            device_options=self._src.device_options,
            reference_clock=self.reference_clock,
            ports=self.ports,
            device_type=convert_device_type(self._src),
            physical_channels=self.physical_channels,
            connections=self.channel_map_entries,
            server=self._server,
            device_class=self._src.device_class,
        )
        return obj


class _InstrumentsLookup:
    """A class for getting new `Instrument` types via old UIDs."""

    def __init__(self, mapping: Mapping[str, Instrument]):
        self._mapping = mapping
        self._pc_by_instrument_lookup = {}
        for instrument in self._mapping.values():
            self._pc_by_instrument_lookup[instrument.uid] = {
                pc.name: pc for pc in instrument.physical_channels
            }

    def get_instrument(self, uid_or_path: str) -> Instrument:
        try:
            return self._mapping[uid_or_path]
        except KeyError:
            return self._mapping[LogicalSignalPhysicalChannelUID(uid_or_path).group]

    def get_physical_channel(self, uid: str) -> PhysicalChannel:
        pc_uid = LogicalSignalPhysicalChannelUID(uid)
        return self._pc_by_instrument_lookup[self._mapping[pc_uid.group].uid][
            pc_uid.name
        ]


def _make_internal_connection_to_same_type_input_port(
    from_instrument: Instrument,
    from_port: Port,
    to_instrument: Instrument,
    to_port_type: PortType,
) -> SetupInternalConnection:
    to_port = None
    for to_instrument_port in to_instrument.ports:
        if to_instrument_port.type == to_port_type:
            to_port = to_instrument_port
            break
    if not to_port:
        raise LabOneQException(
            f"Instrument '{from_instrument.uid}' '{from_port}' has a connection to an"
            f" instrument '{to_instrument.uid}' without '{to_port_type}' ports."
        )
    return SetupInternalConnection(
        from_instrument=from_instrument,
        from_port=from_port,
        to_instrument=to_instrument,
        to_port=to_port,
    )


def make_device_to_device_connections(
    instrument_converters: List[InstrumentConverter],
) -> List[SetupInternalConnection]:
    conns = []
    instrument_lookup = _InstrumentsLookup(
        {s.uid: s.instrument for s in instrument_converters}
    )
    for converter in instrument_converters:
        for (
            logical_signal,
            port,
        ) in converter.connection_converter.connections_from_port_to_remote_logical_signal():
            from_port = converter.port_converter.ports_by_legacy_uid(port.uid)[0]
            remote_ls_pc_uid = logical_signal.physical_channel.uid
            to_instrument = instrument_lookup.get_instrument(remote_ls_pc_uid)
            for to_port in instrument_lookup.get_physical_channel(
                remote_ls_pc_uid
            ).ports:
                connection = SetupInternalConnection(
                    from_instrument=converter.instrument,
                    from_port=from_port,
                    to_instrument=to_instrument,
                    to_port=to_port,
                )
                conns.append(connection)
        for (
            device_uid,
            port,
        ) in converter.connection_converter.connections_from_port_to_another_device():
            from_port = converter.port_converter.ports_by_legacy_uid(port.uid)[0]
            to_instrument = instrument_lookup.get_instrument(device_uid)
            connection = _make_internal_connection_to_same_type_input_port(
                from_instrument=converter.instrument,
                from_port=from_port,
                to_instrument=to_instrument,
                to_port_type=from_port.type,
            )
            conns.append(connection)
    return conns


def combine_shfqa_and_shfsg(
    shfqa: legacy_instruments.SHFQA, shfsg: legacy_instruments.SHFSG
) -> legacy_instruments.SHFQC:
    if shfqa.interface != shfsg.interface:
        raise AssertionError(
            f"Virtual SHFQA {shfqa.uid} and SHFSG {shfsg.uid} have different"
            " interfaces"
        )
    if shfqa.server_uid != shfsg.server_uid:
        raise AssertionError(
            f"Virtual SHFQA {shfqa.uid} and SHFSG {shfsg.uid} have different"
            " server_uids"
        )
    if shfqa.address != shfsg.address:
        raise AssertionError(
            f"Virtual SHFQA {shfqa.uid} and SHFSG {shfsg.uid} have different"
            " addresses"
        )
    if shfqa.reference_clock_source != shfsg.reference_clock_source:
        raise AssertionError(
            f"Virtual SHFQA {shfqa.uid} and SHFSG {shfsg.uid} have different"
            " reference clock sources"
        )

    connections = shfqa.connections + shfsg.connections

    device_options = None
    if shfqa is not None:
        device_options = shfqa.device_options
    if shfsg is not None and shfsg.device_options is not None:
        if device_options is not None:
            assert device_options == shfsg.device_options
        device_options = shfsg.device_options

    return legacy_instruments.SHFQC(
        uid=shfqa.uid,
        interface=shfqa.interface,
        server_uid=shfqa.server_uid,
        address=shfqa.address,
        reference_clock_source=shfqa.reference_clock_source,
        connections=connections,
        device_options=device_options,
    )


def convert_device_setup_to_setup(
    device_setup: legacy_device.device_setup.DeviceSetup,
) -> Setup:
    """Convert legacy `DeviceSetup` into `Setup`."""
    # Note: device_setup.physical_channels is completely ignored.
    #       All the information needed is pulled directly from
    #       the logical signal groups.
    servers = {}
    for name, server in device_setup.servers.items():
        servers[name] = convert_dataserver(
            server, device_setup._server_leader_instrument(name)
        )

    calibration = calibration_converter.convert_calibration(
        device_setup.get_calibration(),
        uid_formatter=calibration_converter.format_ls_pc_uid,
    )

    lsgs, ls_legacy_to_new_map = convert_logical_signal_groups_with_ls_mapping(
        device_setup.logical_signal_groups
    )

    converters = [
        InstrumentConverter(
            instr,
            ls_legacy_to_new_map,
            servers[instr.server_uid],
        )
        for instr in device_setup.instruments
    ]

    instruments = [converter.instrument for converter in converters]

    setup_internal_connections = make_device_to_device_connections(converters)

    return Setup(
        uid=device_setup.uid,
        servers=servers,
        instruments=instruments,
        setup_internal_connections=setup_internal_connections,
        logical_signal_groups=lsgs,
        calibration=calibration,
    )
