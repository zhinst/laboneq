# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import itertools
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

from laboneq.core.exceptions.laboneq_exception import LabOneQException
from laboneq.data.setup_description import (
    Connection,
    DeviceType,
    Instrument,
    IOSignalType,
    LogicalSignal,
    LogicalSignalGroup,
    PhysicalChannel,
    PhysicalChannelType,
    Port,
    Server,
    Setup,
    SetupInternalConnection,
)

_logger = logging.getLogger(__name__)

PATH_SEPARATOR = "/"

LogicalSignalGroups_Path = "logical_signal_groups"
LogicalSignalGroups_Path_Abs = PATH_SEPARATOR + LogicalSignalGroups_Path


# Terminal Symbols
T_HDAWG_DEVICE = "HDAWG"
T_UHFQA_DEVICE = "UHFQA"
T_SHFQA_DEVICE = "SHFQA"
T_SHFSG_DEVICE = "SHFSG"
T_SHFQC_DEVICE = "SHFQC"
T_PQSC_DEVICE = "PQSC"
T_ALL_DEVICE_TYPES = [
    T_HDAWG_DEVICE,
    T_UHFQA_DEVICE,
    T_SHFQA_DEVICE,
    T_SHFSG_DEVICE,
    T_SHFQC_DEVICE,
    T_PQSC_DEVICE,
]
T_UID = "uid"
T_ADDRESS = "address"
T_INTERFACE = "interface"
T_IQ_SIGNAL = "iq_signal"
T_ACQUIRE_SIGNAL = "acquire_signal"
T_RF_SIGNAL = "rf_signal"
T_TO = "to"
T_EXTCLK = "external_clock_signal"
T_INTCLK = "internal_clock_signal"
T_PORT = "port"
T_PORTS = "ports"

SIGNAL_TYPE_DIRECTORY = {
    T_IQ_SIGNAL: IOSignalType.IQ,
    T_ACQUIRE_SIGNAL: IOSignalType.IQ,
    T_RF_SIGNAL: IOSignalType.RF,
    T_TO: IOSignalType.DIO,
}


# Models 'instruments' (former 'instrument_list') part of the descriptor:
#     instruments:
#       HDAWG:
#       - address: DEV8001
#         uid: device_hdawg
#       SHFQA:
#       - address: DEV12001
#         uid: device_shfqa
#       PQSC:
#       - address: DEV10001
#         uid: device_pqsc
InstrumentsType = Dict[str, List[Dict[str, str]]]

# Models 'connections' part of the descriptor:
#     connections:
#       device_hdawg:
#         - iq_signal: q0/drive_line
#           ports: [SIGOUTS/0, SIGOUTS/1]
#         - to: device_uhfqa
#           port: DIOS/0
#       device_uhfqa:
#         - iq_signal: q0/measure_line
#           ports: [SIGOUTS/0, SIGOUTS/1]
#         - acquire_signal: q0/acquire_line
#       device_pqsc:
#         - to: device_hdawg
#           port: ZSYNCS/0
ConnectionsType = Dict[str, List[Dict[str, Union[str, List[str]]]]]

# Models 'dataservers' part of the descriptor:
#     dataservers:
#       zi_server:
#         host: 127.0.0.1
#         port: 8004
#         instruments: [device_hdawg, device_uhfqa, device_pqsc]
DataServersType = Dict[str, Dict[str, Union[str, List[str]]]]


def _iterate_over_descriptors_of_type(instruments: InstrumentsType, device_type: str):
    for descriptor in instruments.get(device_type, []):
        yield descriptor[T_UID], descriptor[T_ADDRESS], descriptor.get(T_INTERFACE)


def _skip_nones(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def _port_decoder(port_desc, additional_switch_keys=None) -> Tuple[str, str, List[str]]:
    if additional_switch_keys is None:
        additional_switch_keys = []
    if isinstance(port_desc, dict):
        port_desc = dict(port_desc)  # make a copy
    else:
        port_desc = {port_desc: None}

    port = port_desc.pop(T_PORT, None)
    ports = port_desc.pop(T_PORTS, None)
    if ports is not None and port is not None:
        raise LabOneQException(
            f"Both ports and port specified, but only one is allowed: {port_desc}"
        )
    if ports is None:
        if port is None:
            local_ports = []
        else:
            local_ports = [port]
    elif isinstance(ports, str):
        local_ports = [ports]
    else:
        local_ports = ports

    signal_keys = [T_IQ_SIGNAL, T_ACQUIRE_SIGNAL, T_RF_SIGNAL]
    trigger_keys = [T_TO]
    path_keys = signal_keys + trigger_keys
    all_keys = path_keys + additional_switch_keys

    signal_type_keyword = None
    remote_path = None
    for key in all_keys:
        if key in port_desc:
            signal_type_keyword = key
            remote_path = port_desc.pop(key)
            break

    if signal_type_keyword is None:
        raise LabOneQException(
            "Missing signal type: Expected one of the following keywords: "
            + ", ".join(all_keys)
        )
    if signal_type_keyword in path_keys and not remote_path:
        raise LabOneQException(
            f"Missing path: specify '{signal_type_keyword}: <group>/<line>'"
        )

    if signal_type_keyword in signal_keys:
        remote_path = PATH_SEPARATOR.join(["", "logical_signal_groups", remote_path])

    if port_desc:
        raise LabOneQException(f"Unknown keyword found: {list(port_desc.keys())[0]}")

    return signal_type_keyword, remote_path, local_ports


def _path_to_signal(path):
    if PATH_SEPARATOR in path:
        split_path = path.split(PATH_SEPARATOR)
        if split_path[1] == LogicalSignalGroups_Path:
            return split_path[2], split_path[3]
        else:
            return split_path[0], split_path[1]
    return None


def _create_physical_channel(
    ports: List[str], signal_type_token: str, device_id, physical_signals
) -> Optional[PhysicalChannel]:
    if signal_type_token in (T_IQ_SIGNAL, T_ACQUIRE_SIGNAL):
        channel_type = PhysicalChannelType.IQ_CHANNEL
    elif signal_type_token == T_RF_SIGNAL:
        channel_type = PhysicalChannelType.RF_CHANNEL
    else:
        return None

    split_ports = [port.split(PATH_SEPARATOR) for port in ports]
    signal_name = "_".join(
        (
            group[0]
            for group in itertools.groupby([x for y in zip(*split_ports) for x in y])
        )
    ).lower()

    if device_id not in physical_signals:
        physical_signals[device_id] = []
    else:
        other_signal: PhysicalChannel = next(
            (ps for ps in physical_signals[device_id] if ps.uid == signal_name), None
        )
        if other_signal is not None:
            return other_signal

    physical_channel = PhysicalChannel(uid=f"{signal_name}", type=channel_type)
    physical_signals[device_id].append(physical_channel)
    return physical_channel


class DeviceSetupGenerator:
    @staticmethod
    def from_descriptor(
        yaml_text: str,
        server_host: str = None,
        server_port: str = None,
        setup_name: str = None,
    ):
        from yaml import load

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        setup_desc = load(yaml_text, Loader=Loader)

        return DeviceSetupGenerator.from_dicts(
            instrument_list=setup_desc.get("instrument_list"),
            instruments=setup_desc.get("instruments"),
            connections=setup_desc.get("connections"),
            dataservers=setup_desc.get("dataservers"),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_desc.get("setup_name")
            if setup_name is None
            else setup_name,
        )

    @staticmethod
    def from_yaml(
        filepath,
        server_host: str = None,
        server_port: str = None,
        setup_name: str = None,
    ):
        from yaml import load

        try:
            from yaml import CLoader as Loader
        except ImportError:
            from yaml import Loader

        with open(filepath) as fp:
            setup_desc = load(fp, Loader=Loader)

        return DeviceSetupGenerator.from_dicts(
            instrument_list=setup_desc.get("instrument_list"),
            instruments=setup_desc.get("instruments"),
            connections=setup_desc.get("connections"),
            dataservers=setup_desc.get("dataservers"),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_desc.get("setup_name")
            if setup_name is None
            else setup_name,
        )

    @staticmethod
    def from_dicts(
        instrument_list: InstrumentsType = None,
        instruments: InstrumentsType = None,
        connections: ConnectionsType = None,
        dataservers: DataServersType = None,
        server_host: str = None,
        server_port: str = None,
        setup_name: str = None,
    ):
        if instrument_list is not None:
            if instruments is None:
                warnings.warn(
                    "'instrument_list' section is deprecated in setup descriptor, use 'instruments' instead.",
                    FutureWarning,
                )
                instruments = instrument_list
            else:
                warnings.warn(
                    "Both 'instrument_list' and 'instruments' are present in the setup descriptor, deprecated 'instrument_list' ignored.",
                    FutureWarning,
                )

        if instruments is None:
            raise LabOneQException(
                "'instruments' section is mandatory in the setup descriptor."
            )

        if connections is None:
            connections = {}

        if setup_name is None:
            setup_name = "unknown"

        if server_host is not None:
            if dataservers is not None:
                _logger.warning(
                    "Servers definition in the descriptor will be overridden by the server passed to the constructor."
                )
            dataservers = {
                "zi_server": {
                    "host": server_host,
                    "port": "8004" if server_port is None else server_port,
                    "instruments": [],
                }
            }

        if dataservers is None:
            raise LabOneQException(
                "At least one server must be defined either in the descriptor or in the constructor."
            )

        # Construct servers
        servers = [
            (
                Server(
                    uid=server_uid,
                    host=server_def["host"],
                    port=server_def.get("port", 8004),
                    api_level=6,
                ),
                server_def.get("instruments", []),
            )
            for server_uid, server_def in dataservers.items()
        ]

        # TODO(2K): Remove, leader device must be determined from connections
        # Keeping this only to satisfy the tests
        if len(servers) == 1:
            pqsc_dicts = instruments.get(T_PQSC_DEVICE, [])
            if len(pqsc_dicts) > 0:
                servers[0][0].leader_uid = pqsc_dicts[0][T_UID]

        def server_finder(device_uid: str) -> str:
            default_data_server: Server = None
            explicit_data_server: Server = None
            for data_server, devices in servers:
                if default_data_server is None and len(devices) == 0:
                    default_data_server = data_server
                if device_uid in devices:
                    if explicit_data_server is not None:
                        raise LabOneQException(
                            f"Device '{device_uid}' assigned to multiple data servers: "
                            f"[{explicit_data_server.uid}, {data_server.uid}]."
                        )
                    explicit_data_server = data_server
            if explicit_data_server is not None:
                return explicit_data_server
            if default_data_server is None:
                raise LabOneQException(
                    f"Couldn't determine the data server for the device '{device_uid}'."
                )
            return default_data_server

        # Define instruments
        out_instruments: List[Instrument] = []
        for it, il in {**instrument_list, **instruments}.items():
            for instrument_def in il:
                instrument = Instrument(
                    uid=instrument_def[T_UID],
                    device_type=DeviceType[it],
                    server=server_finder(instrument_def[T_UID]),
                    address=instrument_def.get(T_ADDRESS, ""),
                )
                out_instruments.append(instrument)

        instruments_by_uid = {i.uid: i for i in out_instruments}

        logical_signals_candidates = []
        logical_signal_groups = []
        physical_signals = {}
        setup_internal_connections = []

        for device_uid, conns in connections.items():
            instrument = instruments_by_uid[device_uid]
            for conn in conns:
                signal_type_keyword, remote_path, local_ports = _port_decoder(conn)
                if PATH_SEPARATOR in remote_path:
                    logical_signals_candidates.append(
                        {
                            "lsg_uid": remote_path.split(PATH_SEPARATOR)[2],
                            "signal_id": remote_path.split(PATH_SEPARATOR)[3],
                        }
                    )

        def ls_path_from_parts(lsg_uid, signal_id):
            return f"{LogicalSignalGroups_Path_Abs}/{lsg_uid}/{signal_id}"

        ls_by_path = {}
        logical_signal_group_uids = set(
            [ls["lsg_uid"] for ls in logical_signals_candidates]
        )
        for lsg_uid in logical_signal_group_uids:
            signals = [
                LogicalSignal(
                    uid=f"{ls['signal_id']}",
                    name=ls["signal_id"],
                    path=ls_path_from_parts(lsg_uid, ls["signal_id"]),
                )
                for ls in logical_signals_candidates
                if ls["lsg_uid"] == lsg_uid
            ]
            ls_by_path = {**ls_by_path, **{ls.path: ls for ls in signals}}

            logical_signal_groups.append(LogicalSignalGroup(lsg_uid, signals))

        for lsg in logical_signal_groups:
            lsg.logical_signals = {ls.uid: ls for ls in lsg.logical_signals}
        logical_signal_groups = {lsg.uid: lsg for lsg in logical_signal_groups}

        # Define connections
        for device_uid, conns in connections.items():
            physical_channels_by_uid = {}
            instrument = instruments_by_uid[device_uid]
            for conn in conns:
                signal_type_keyword, remote_path, local_ports = _port_decoder(conn)
                if signal_type_keyword == T_ACQUIRE_SIGNAL:
                    if instrument.device_type == DeviceType.UHFQA:
                        local_ports = ["QAS/0", "QAS/1"]

                logical_signal = None
                if PATH_SEPARATOR in remote_path:
                    logical_signal_id = {
                        "lsg_uid": remote_path.split(PATH_SEPARATOR)[2],
                        "signal_id": remote_path.split(PATH_SEPARATOR)[3],
                    }
                    logical_signal = ls_by_path[ls_path_from_parts(**logical_signal_id)]
                physical_channel = _create_physical_channel(
                    local_ports, signal_type_keyword, device_uid, physical_signals
                )

                if physical_channel is not None:
                    if physical_channel.uid in physical_channels_by_uid:
                        physical_channel = physical_channels_by_uid[
                            physical_channel.uid
                        ]
                    else:
                        physical_channels_by_uid[
                            physical_channel.uid
                        ] = physical_channel
                        instrument.physical_channels.append(physical_channel)

                if logical_signal is not None:
                    instrument.connections.append(
                        Connection(
                            logical_signal=logical_signal,
                            physical_channel=physical_channel,
                        )
                    )
                for i, p in enumerate(local_ports):
                    if p not in [port.path for port in instrument.ports]:
                        current_port = Port(path=p, physical_channel=physical_channel)
                        instrument.ports.append(current_port)
                    else:
                        current_port = next(
                            port for port in instrument.ports if port.path == p
                        )

                    if signal_type_keyword == T_TO:
                        setup_internal_connections.append(
                            SetupInternalConnection(
                                from_instrument=instrument,
                                from_port=current_port,
                                to_instrument=instruments_by_uid[remote_path],
                            )
                        )

        servers = {s.uid: s for s, _ in servers}
        device_setup_constructor_args = {
            "uid": setup_name,
            "servers": servers,
            "instruments": out_instruments,
            "logical_signal_groups": logical_signal_groups,
            "setup_internal_connections": setup_internal_connections,
        }

        return Setup(**device_setup_constructor_args)
