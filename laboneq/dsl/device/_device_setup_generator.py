# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple, Union

import jsonschema
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

import laboneq.core.path as qct_path
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import ReferenceClockSource
from laboneq.dsl.device import _device_setup_modifier as modifier
from laboneq.dsl.device.connection import InternalConnection, SignalConnection
from laboneq.dsl.device.instruments import (
    HDAWG,
    PQSC,
    SHFPPC,
    SHFQA,
    SHFQC,
    SHFSG,
    PRETTYPRINTERDEVICE,
    UHFQA,
    NonQC,
)
from laboneq.dsl.device.servers import DataServer

if TYPE_CHECKING:
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.logical_signal_group import LogicalSignalGroup
    from laboneq.dsl.quantum import QuantumElement


_logger = logging.getLogger(__name__)


# Terminal Symbols
T_HDAWG_DEVICE = "HDAWG"
T_PRETTYPRINTER_DEVICE = "PRETTYPRINTERDEVICE"
T_UHFQA_DEVICE = "UHFQA"
T_SHFQA_DEVICE = "SHFQA"
T_SHFSG_DEVICE = "SHFSG"
T_SHFQC_DEVICE = "SHFQC"
T_SHFPPC_DEVICE = "SHFPPC"
T_PQSC_DEVICE = "PQSC"
T_ALL_DEVICE_TYPES = [
    T_HDAWG_DEVICE,
    T_PRETTYPRINTER_DEVICE,
    T_UHFQA_DEVICE,
    T_SHFQA_DEVICE,
    T_SHFSG_DEVICE,
    T_SHFQC_DEVICE,
    T_SHFPPC_DEVICE,
    T_PQSC_DEVICE,
]
T_UID = "uid"
T_ADDRESS = "address"
T_OPTIONS = "options"
T_INTERFACE = "interface"
T_IQ_SIGNAL = "iq_signal"
T_ACQUIRE_SIGNAL = "acquire_signal"
T_RF_SIGNAL = "rf_signal"
T_TO = "to"
T_EXTCLK = "external_clock_signal"
T_INTCLK = "internal_clock_signal"
T_PORT = "port"
T_PORTS = "ports"


# Models 'instruments' (former 'instrument_list') part of the descriptor:
#     instruments:
#       HDAWG:
#       - address: DEV8001
#         uid: device_hdawg
#         options: HDAWG8/CNT/ME/MF
#       SHFQA:
#       - address: DEV12001
#         uid: device_shfqa
#         options: SHFQA2
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
        yield (
            descriptor[T_UID],
            descriptor[T_ADDRESS],
            descriptor.get(T_INTERFACE),
            descriptor.get(T_OPTIONS),
        )


def _skip_nones(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


def _make_ppc_connection(_: str, remote_path: str, local_ports: str):
    # In the descriptor SHFPPC has `T_TO` signal type keyword, but is actually a signal connection without a type.
    return SignalConnection(
        uid=qct_path.remove_logical_signal_prefix(remote_path),
        type=None,
        ports=local_ports,
    )


def _make_connection(signal_type_kw: str, remote_path: str, local_ports: str):
    SIGTYPE_T_TO_CONNECTION_TYPE = {
        T_IQ_SIGNAL: "iq",
        T_RF_SIGNAL: "rf",
        T_ACQUIRE_SIGNAL: "acquire",
    }
    if signal_type_kw == T_TO:
        conn = InternalConnection(to=remote_path, from_port=local_ports)
    else:
        conn = SignalConnection(
            uid=qct_path.remove_logical_signal_prefix(remote_path),
            type=SIGTYPE_T_TO_CONNECTION_TYPE[signal_type_kw],
            ports=local_ports,
        )
    return conn


def _is_valid_remote_path(path: str) -> bool:
    if len(path.split(qct_path.Separator)) != 2:
        return False
    if not all(path.split(qct_path.Separator)):
        return False
    return True


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
    trigger_keys = []
    trigger_keys.append(T_TO)
    path_keys = signal_keys + trigger_keys
    all_keys = path_keys + additional_switch_keys

    signal_type_keyword = None
    remote_path = None
    for key in all_keys:
        if key in port_desc:
            signal_type_keyword = key
            remote_path = port_desc.pop(key)
            break
    if port_desc:
        raise LabOneQException(f"Unknown keyword found: {list(port_desc.keys())[0]}")
    if signal_type_keyword is None:
        raise LabOneQException(
            "Missing signal type: Expected one of the following keywords: "
            + ", ".join(all_keys)
        )
    if signal_type_keyword in path_keys and not remote_path:
        raise LabOneQException(
            f"Missing path: specify '{signal_type_keyword}: <group>{qct_path.Separator}<line>'"
        )
    if signal_type_keyword in signal_keys:
        if not _is_valid_remote_path(remote_path):
            raise LabOneQException(
                f"Invalid path: specify '{signal_type_keyword}: <group>{qct_path.Separator}<line>'"
            )
        remote_path = qct_path.insert_logical_signal_prefix(remote_path)
    return signal_type_keyword, remote_path, local_ports


def make_zi_devices(
    instruments: InstrumentsType,
    connections: ConnectionsType,
    server_finder: Callable[[str], str],
    setup: DeviceSetup,
):
    zi_instruments = {
        # Descriptor key: Intrument class, Reference clock option, connection builder.
        T_HDAWG_DEVICE: (HDAWG, T_EXTCLK, _make_connection),
        T_UHFQA_DEVICE: (UHFQA, T_EXTCLK, _make_connection),
        T_SHFQA_DEVICE: (SHFQA, T_EXTCLK, _make_connection),
        T_SHFSG_DEVICE: (SHFSG, T_EXTCLK, _make_connection),
        T_SHFQC_DEVICE: (SHFQC, T_EXTCLK, _make_connection),
        T_PRETTYPRINTER_DEVICE: (PRETTYPRINTERDEVICE, T_EXTCLK, _make_connection),
        T_SHFPPC_DEVICE: (SHFPPC, T_EXTCLK, _make_ppc_connection),
        T_PQSC_DEVICE: (PQSC, T_INTCLK, _make_connection),
    }
    ref_clk_types = {
        T_EXTCLK: ReferenceClockSource.EXTERNAL,
        T_INTCLK: ReferenceClockSource.INTERNAL,
    }
    for desc_key, (dev_cls, ref_clk, make_connection) in zi_instruments.items():
        for uid, address, interface, options in _iterate_over_descriptors_of_type(
            instruments, desc_key
        ):
            dev = dev_cls(
                **_skip_nones(
                    uid=uid,
                    server_uid=server_finder(uid),
                    address=address,
                    interface=interface,
                    device_options=options,
                )
            )
            conns = connections.get(uid, [])
            if ref_clk in conns:
                dev.reference_clock_source = ref_clk_types[ref_clk]
                conns.remove(ref_clk)
            modifier.add_instrument(setup, dev)
            for port_desc in conns:
                signal_type_keyword, remote_path, local_ports = _port_decoder(port_desc)
                try:
                    modifier.add_connection(
                        setup,
                        dev.uid,
                        make_connection(signal_type_keyword, remote_path, local_ports),
                    )
                except (LabOneQException, ValueError) as e:
                    msg = f"Error '{uid}' ({signal_type_keyword}, {qct_path.remove_logical_signal_prefix(remote_path)}): "
                    msg += str(e)
                    raise LabOneQException(msg) from e


def make_unmanaged_instrument(
    instruments: InstrumentsType,
    server_finder: Callable[[str], str],
    setup: DeviceSetup,
):
    for dev_type, devices in instruments.items():
        if dev_type not in T_ALL_DEVICE_TYPES:
            for descriptor in devices:
                uid = descriptor[T_UID]
                dev = NonQC(
                    **_skip_nones(
                        server_uid=server_finder(uid),
                        uid=uid,
                        address=descriptor[T_ADDRESS],
                        interface=descriptor.get(T_INTERFACE),
                        dev_type=dev_type,
                    )
                )
                modifier.add_instrument(setup, dev)


def make_qubits(
    qubit_descriptor: list[dict],
    logical_signal_groups: list[LogicalSignalGroup],
    types: dict[str, QuantumElement],
) -> dict[str, QuantumElement]:
    """Make qubits from their descriptor in Device Setup descriptor.

    Args:
        qubit_descriptor: `qubits` section the descriptor.
        logical_signal_groups: Logical signal groups.
        types: Mapping of types.
            Mapping keys are used to select the correct `QuantumElement` to make
            qubit `type`.

    Returns:
        Dictionary of `QuantumElements`, with their `uid` as key.
            The keys are sorted.

    Raises:
        LabOneQException: If the qubits are defined incorrectly.
    """
    if not isinstance(qubit_descriptor, list):
        msg = "Invalid 'qubits' definition: Must be a list of qubits."
        raise LabOneQException(msg)
    schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": ["string", "array"],
                "items": {
                    "type": "string",
                },
                "uniqueItems": True,
            },
            "type": {"type": "string"},
        },
        "required": ["name", "type"],
        "additionalProperties": False,
    }
    qubits = {}
    for desc in qubit_descriptor:
        try:
            jsonschema.validate(desc, schema=schema)
        except jsonschema.exceptions.ValidationError as error:
            msg = error.message
            raise LabOneQException(f"Invalid 'qubit' definition: {msg}") from error
        q_type = desc["type"]
        try:
            quantum_element = types[q_type]
        except KeyError:
            msg = f"'type': '{q_type}' not one of {list(types.keys())}."
            raise LabOneQException(f"Invalid 'qubit' definition: {msg}") from None
        q_defs = desc["name"] if isinstance(desc["name"], list) else [desc["name"]]
        for q_def in q_defs:
            if q_def in qubits:
                msg = f"Qubit '{q_def}' has multiple definitions."
                raise LabOneQException(f"Invalid 'qubit' definition: {msg}")
            q_type = desc["type"]
            try:
                lsg = logical_signal_groups[q_def]
            except KeyError:
                msg = f"Qubit '{q_def}' has no connections."
                raise LabOneQException(f"Invalid 'qubit' definition: {msg}") from None
            qubits[q_def] = quantum_element.from_logical_signal_group(q_def, lsg=lsg)
    return {key: qubits[key] for key in sorted(qubits)}


class _DeviceSetupGenerator:
    @staticmethod
    def from_descriptor(
        yaml_text: str,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ):
        setup_desc = load(yaml_text, Loader=Loader)

        return _DeviceSetupGenerator.from_dicts(
            instrument_list=setup_desc.get("instrument_list"),
            instruments=setup_desc.get("instruments"),
            connections=setup_desc.get("connections"),
            dataservers=setup_desc.get("dataservers"),
            qubits=setup_desc.get("qubits", []),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_desc.get("setup_name")
            if setup_name is None
            else setup_name,
        )

    @staticmethod
    def from_yaml(
        filepath,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ):
        with open(filepath) as fp:
            setup_desc = load(fp, Loader=Loader)

        return _DeviceSetupGenerator.from_dicts(
            instrument_list=setup_desc.get("instrument_list"),
            instruments=setup_desc.get("instruments"),
            connections=setup_desc.get("connections"),
            dataservers=setup_desc.get("dataservers"),
            qubits=setup_desc.get("qubits", []),
            server_host=server_host,
            server_port=server_port,
            setup_name=setup_desc.get("setup_name")
            if setup_name is None
            else setup_name,
        )

    @staticmethod
    def from_dicts(
        instrument_list: InstrumentsType | None = None,
        instruments: InstrumentsType | None = None,
        connections: ConnectionsType | None = None,
        dataservers: DataServersType | None = None,
        qubits: list | None = None,
        server_host: str | None = None,
        server_port: str | None = None,
        setup_name: str | None = None,
    ):
        # To avoid circular imports
        from laboneq.dsl import quantum
        from laboneq.dsl.device.device_setup import DeviceSetup

        if qubits is None:
            qubits = []
        if instrument_list is not None:
            if instruments is None:
                warnings.warn(
                    "'instrument_list' section is deprecated in setup descriptor, use 'instruments' instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                instruments = instrument_list
            else:
                warnings.warn(
                    "Both 'instrument_list' and 'instruments' are present in the setup descriptor, deprecated 'instrument_list' ignored.",
                    FutureWarning,
                    stacklevel=2,
                )

        if instruments is None:
            raise LabOneQException(
                "'instruments' section is mandatory in the setup descriptor."
            )

        if connections is None:
            connections = {}

        if server_host is not None:
            if dataservers is not None:
                _logger.warning(
                    "Servers definition in the descriptor will be overridden by the server passed to the constructor."
                )
            dataservers = {
                "zi_server": {
                    "host": server_host,
                    "port": "8004" if server_port is None else server_port,
                }
            }

        if dataservers is None:
            raise LabOneQException(
                "At least one server must be defined either in the descriptor or in the constructor."
            )

        # Construct servers
        servers: List[Tuple[DataServer, List[str]]] = [
            (
                DataServer(
                    uid=server_uid,
                    host=server_def["host"],
                    port=server_def.get("port", 8004),
                    api_level=6,
                ),
                server_def.get("instruments", []),
            )
            for server_uid, server_def in dataservers.items()
        ]

        def server_finder(device_uid: str) -> str:
            default_data_server: DataServer = None
            explicit_data_server: DataServer = None
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
                return explicit_data_server.uid
            if default_data_server is None:
                raise LabOneQException(
                    f"Couldn't determine the data server for the device '{device_uid}'."
                )
            return default_data_server.uid

        setup = DeviceSetup(**_skip_nones(uid=setup_name))
        for server in servers:
            dataserver = server[0]
            setup.add_dataserver(
                host=dataserver.host,
                port=dataserver.port,
                api_level=dataserver.api_level,
                uid=dataserver.uid,
            )
        make_zi_devices(instruments, connections, server_finder, setup)
        make_unmanaged_instrument(instruments, server_finder, setup)
        setup.qubits = make_qubits(
            qubits,
            setup.logical_signal_groups,
            types={
                "qubit": quantum.Qubit,
                "transmon": quantum.Transmon,
            },
        )
        return setup
