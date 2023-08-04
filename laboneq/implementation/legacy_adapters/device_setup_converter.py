# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Mapping, Optional, Tuple

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
from laboneq.implementation.legacy_adapters import calibration_converter
from laboneq.implementation.legacy_adapters.utils import (
    LogicalSignalPhysicalChannelUID,
    raise_not_implemented,
)

if TYPE_CHECKING:
    from laboneq.dsl.device import logical_signal_group as legacy_lsg
    from laboneq.dsl.device import servers as legacy_servers
    from laboneq.dsl.device.instruments.zi_standard_instrument import (
        ZIStandardInstrument,
    )


def legacy_instrument_ports(device_type: DeviceType) -> List[legacy_device.Port]:
    if device_type == DeviceType.HDAWG:
        return legacy_instruments.HDAWG().ports
    if device_type == DeviceType.SHFQA:
        return legacy_instruments.SHFQA().ports
    if device_type == DeviceType.PQSC:
        return legacy_instruments.PQSC().ports
    if device_type == DeviceType.UHFQA:
        return legacy_instruments.UHFQA().ports
    if device_type == DeviceType.SHFSG:
        return legacy_instruments.SHFSG().ports
    if device_type == DeviceType.SHFQC:
        return legacy_instruments.SHFSG().ports + legacy_instruments.SHFQA().ports
    if device_type == DeviceType.SHFPPC:
        return legacy_instruments.SHFPPC().ports
    if device_type == DeviceType.UNMANAGED:
        return legacy_instruments.NonQC().ports
    raise_not_implemented(device_type)


def legacy_signal_to_port_type(
    signal: Optional[legacy_enums.IOSignalType],
) -> Optional[PortType]:
    if signal is None:
        return None
    if not isinstance(signal, legacy_enums.IOSignalType):
        raise_not_implemented(signal)
    if signal == legacy_enums.IOSignalType.DIO:
        return PortType.DIO
    if signal == legacy_enums.IOSignalType.ZSYNC:
        return PortType.ZSYNC
    return PortType.RF


def convert_instrument_port(legacy: legacy_device.Port) -> Port:
    return Port(path=legacy.uid, type=legacy_signal_to_port_type(legacy.signal_type))


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


def convert_dataserver(target: legacy_servers.DataServer) -> Server:
    return Server(
        uid=target.uid,
        api_level=target.api_level,
        host=target.host,
        leader_uid=target.leader_uid,
        port=int(target.port),
    )


def convert_reference_clock_source(
    target: Optional[legacy_enums.ReferenceClockSource],
) -> Optional[ReferenceClockSource]:
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
    if isinstance(target, legacy_instruments.PQSC):
        return DeviceType.PQSC
    if isinstance(target, legacy_instruments.UHFQA):
        return DeviceType.UHFQA
    if isinstance(target, legacy_instruments.SHFPPC):
        return DeviceType.SHFPPC
    if isinstance(target, legacy_instruments.SHFSG):
        return DeviceType.SHFSG
    if isinstance(target, legacy_instruments.NonQC):
        return DeviceType.UNMANAGED
    raise_not_implemented(target)


def convert_physical_channel_type(
    target: Optional[legacy_io_units.PhysicalChannelType],
) -> Optional[PhysicalChannelType]:
    if target is None:
        return None
    if target == legacy_io_units.PhysicalChannelType.IQ_CHANNEL:
        return PhysicalChannelType.IQ_CHANNEL
    if target == legacy_io_units.PhysicalChannelType.RF_CHANNEL:
        return PhysicalChannelType.RF_CHANNEL
    raise_not_implemented(target)


from laboneq.implementation.legacy_adapters import utils


def convert_physical_channel(
    target: legacy_io_units.PhysicalChannel,
    ports: Optional[Iterable[legacy_device.Port]] = None,
) -> PhysicalChannel:
    if ports is None:
        ports = []

    pc_helper = utils.LogicalSignalPhysicalChannelUID(target.uid)

    return PhysicalChannel(
        name=pc_helper.name,
        group=pc_helper.group,
        type=convert_physical_channel_type(target.type),
        ports=[convert_instrument_port(port) for port in ports],
    )


def _make_converted_lookup(ports: Iterable[legacy_device.Port]) -> Mapping[str, Port]:
    return {
        port.path: port for port in [convert_instrument_port(port) for port in ports]
    }


def make_logical_signal_to_physical_signal_connections(
    connections: List[legacy_device.Connection],
    ls_legacy_new_map: Mapping[legacy_lsg.LogicalSignal, LogicalSignal],
    ports: Iterable[legacy_device.Port],
) -> Tuple[List[ChannelMapEntry], List[PhysicalChannel], List[Port]]:
    logical_signal_lookup = {ls.path: ls for ls in ls_legacy_new_map.keys()}
    port_lookup = {port.uid: port for port in ports}
    ports_converted = _make_converted_lookup(ports)
    physical_chs = {}
    channel_map_entries = []
    # Find connected logical signals and to which ports they point to.
    for conn in connections:
        if logical_signal := logical_signal_lookup.get(conn.remote_path):
            if port := port_lookup.get(conn.local_port):
                # Port and LogicalSignal point to the same direction, should be the same device.
                if port.direction == logical_signal.direction:
                    phys_ch_uid = logical_signal.physical_channel.uid
                    if phys_ch_uid not in physical_chs:
                        physical_ch = convert_physical_channel(
                            logical_signal.physical_channel
                        )
                        physical_ch.direction = convert_io_direction(
                            logical_signal.direction
                        )
                        physical_chs[phys_ch_uid] = physical_ch

                    entry = ChannelMapEntry(
                        physical_channel=physical_chs[phys_ch_uid],
                        logical_signal=ls_legacy_new_map[logical_signal],
                    )
                    if entry not in channel_map_entries:
                        channel_map_entries.append(entry)

                    if ports_converted[port.uid] not in physical_chs[phys_ch_uid].ports:
                        physical_chs[phys_ch_uid].ports.append(
                            ports_converted[port.uid]
                        )
    return (
        channel_map_entries,
        list(physical_chs.values()),
        list(ports_converted.values()),
    )


def convert_instrument(
    target: ZIStandardInstrument,
    ls_legacy_new_map: Mapping[legacy_lsg.LogicalSignal, LogicalSignal],
    server: Server,
) -> Instrument:
    # TODO: How to handle SHFQA / SHFSG with `is_qc=True`
    ref_clk = ReferenceClock()
    if target.reference_clock_source is not None:
        ref_clk.source = convert_reference_clock_source(target.reference_clock_source)
    # Only PQSC has reference clock frequency
    if isinstance(target, legacy_instruments.PQSC):
        ref_clk.frequency = target.reference_clock
    (
        connections,
        physical_channels,
        ports,
    ) = make_logical_signal_to_physical_signal_connections(
        target.connections, ls_legacy_new_map, target.ports
    )

    obj = Instrument(
        uid=target.uid,
        address=target.address,
        interface=target.interface,
        reference_clock=ref_clk,
        ports=ports,
        device_type=convert_device_type(target),
        physical_channels=physical_channels,
        connections=connections,
        server=server,
    )
    return obj


class _InstrumentsLookup:
    """A class for getting new `Instrument` types via old UIDs."""

    def __init__(self, mapping: Mapping[str, Instrument]):
        self._mapping = mapping
        self._pc_by_instrument_lookup = {}
        self._port_by_instrument_lookup = {}
        for instrument in self._mapping.values():
            self._pc_by_instrument_lookup[instrument.uid] = {
                pc.name: pc for pc in instrument.physical_channels
            }
            self._port_by_instrument_lookup[instrument.uid] = {
                port.path: port for port in instrument.ports
            }

    def get_instrument(self, uid: str) -> Instrument:
        try:
            return self._mapping[uid]
        except KeyError:
            return self._mapping[LogicalSignalPhysicalChannelUID(uid).group]

    def get_physical_channel(self, uid: str) -> PhysicalChannel:
        pc_uid = LogicalSignalPhysicalChannelUID(uid)
        return self._pc_by_instrument_lookup[self._mapping[pc_uid.group].uid][
            pc_uid.name
        ]

    def get_instrument_port(self, uid: str, port: str) -> Port:
        return self._port_by_instrument_lookup[self._mapping[uid].uid][port]


def _make_internal_connection_to_same_type_input_port(
    from_instrument, from_port, to_instrument, to_port_type: PortType
):
    to_port = None
    for to_instrument_port in to_instrument.ports:
        if to_instrument_port.type == to_port_type:
            to_port = to_instrument_port
            break
    if not to_port:
        raise LabOneQException(
            f"Instrument '{from_instrument.uid}' '{from_port}' has a connection to an instrument '{to_instrument.uid}' without '{to_port_type}' ports."
        )
    return SetupInternalConnection(
        from_instrument=from_instrument,
        from_port=from_port,
        to_instrument=to_instrument,
        to_port=to_port,
    )


def _make_device_to_device_connections(
    legacy_instruments: List[legacy_device.Instrument],
    legacy_logical_signals: List[legacy_lsg.LogicalSignal],
    instrument_lookup: _InstrumentsLookup,
) -> List[SetupInternalConnection]:
    legacy_ls_lookup = {ls.path: ls for ls in legacy_logical_signals}
    conns = []
    for instr in legacy_instruments:
        from_instrument = instrument_lookup.get_instrument(instr.uid)
        for conn in instr.connections:
            remote_logical_signal = legacy_ls_lookup.get(conn.remote_path)
            from_port = instrument_lookup.get_instrument_port(
                instr.uid, conn.local_port
            )
            # Port is defined to go to logical signal.
            if remote_logical_signal:
                remote_ls_pc_uid = remote_logical_signal.physical_channel.uid
                to_instrument = instrument_lookup.get_instrument(remote_ls_pc_uid)
                # Logical signal and connection point to the different direction: Device to device connection.
                if remote_logical_signal.direction != conn.direction:
                    for to_port in instrument_lookup.get_physical_channel(
                        remote_ls_pc_uid
                    ).ports:
                        conns.append(
                            SetupInternalConnection(
                                from_instrument=from_instrument,
                                from_port=from_port,
                                to_instrument=to_instrument,
                                to_port=to_port,
                            )
                        )
                else:
                    if from_instrument.address != to_instrument.address:
                        from_instr_display = f"{instr.uid}/{conn.local_port}"
                        to_instr_display = f"{remote_ls_pc_uid}"
                        raise LabOneQException(
                            "Instrument to instrument connection ports point to the same direction: "
                            f"{from_instr_display} -> {remote_logical_signal.uid} -> {to_instr_display}"
                        )
            else:
                # Port is defined to go straight to device.
                # Most commonly ZSYNC / DIO input connections.
                if from_port.type in {PortType.ZSYNC, PortType.DIO}:
                    to_instrument = instrument_lookup.get_instrument(conn.remote_path)
                    c = _make_internal_connection_to_same_type_input_port(
                        from_instrument=from_instrument,
                        from_port=from_port,
                        to_instrument=to_instrument,
                        to_port_type=from_port.type,
                    )
                    conns.append(c)
                else:
                    raise NotImplementedError(
                        f"No known input port type in device {to_instrument.uid} for port type {from_port.type}."
                    )
    return conns


def convert_device_setup_to_setup(
    device_setup: legacy_device.device_setup.DeviceSetup,
) -> Setup:
    """Convert legacy `DeviceSetup` into `Setup`."""
    setup = Setup(
        uid=device_setup.uid,
    )
    for name, server in device_setup.servers.items():
        setup.servers[name] = convert_dataserver(server)

    setup.calibration = calibration_converter.convert_calibration(
        device_setup.get_calibration(),
        uid_formatter=calibration_converter.format_ls_pc_uid,
    )
    lsgs, ls_legacy_to_new_map = convert_logical_signal_groups_with_ls_mapping(
        device_setup.logical_signal_groups
    )
    setup.logical_signal_groups = lsgs

    legacy_to_new_instrument_map = {}
    for legacy_instr in device_setup.instruments:
        instr = convert_instrument(
            legacy_instr,
            ls_legacy_to_new_map,
            setup.servers[legacy_instr.server_uid],
        )
        setup.instruments.append(instr)
        legacy_to_new_instrument_map[legacy_instr.uid] = instr
    setup.setup_internal_connections = _make_device_to_device_connections(
        device_setup.instruments,
        ls_legacy_to_new_map.keys(),
        _InstrumentsLookup(legacy_to_new_instrument_map),
    )
    return setup
