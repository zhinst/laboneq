# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Union

from yaml import safe_load

import laboneq.core.path as qct_path
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import ReferenceClockSource
from laboneq.dsl.device import _device_setup_modifier as modifier
from laboneq.dsl.device import instrument_registry as registry
from laboneq.dsl.device.connection import InternalConnection, SignalConnection
from laboneq.dsl.device.servers import DataServer

if TYPE_CHECKING:
    from laboneq.data.instrument_topology import InstrumentTopology
    from laboneq.dsl.device.device_setup import DeviceSetup
    from laboneq.dsl.device.instrument import Instrument
    from laboneq.dsl.device.logical_signal_group import LogicalSignalGroup
    from laboneq.dsl.quantum import QuantumElement


_logger = logging.getLogger(__name__)


# Terminal Symbols
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

_SIGNAL_TYPE_TO_CONNECTION_TYPE = {
    T_IQ_SIGNAL: "iq",
    T_RF_SIGNAL: "rf",
    T_ACQUIRE_SIGNAL: "acquire",
}


# Models 'instruments' part of the descriptor:
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
InstrumentsType = dict[str, list[dict[str, str]]]

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
ConnectionsType = dict[str, list[dict[str, Union[str, list[str]]]]]

# Models 'dataservers' part of the descriptor:
#     dataservers:
#       zi_server:
#         host: 127.0.0.1
#         port: 8004
#         instruments: [device_hdawg, device_uhfqa, device_pqsc]
DataServersType = dict[str, dict[str, Union[str, list[str]]]]


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


def _is_valid_remote_path(path: str) -> bool:
    if len(path.split(qct_path.Separator)) != 2:
        return False
    if not all(path.split(qct_path.Separator)):
        return False
    return True


def _port_decoder(
    port_desc: dict[str, str | list[str]],
) -> tuple[str, str, list[str]]:
    """Decode one entry of a descriptor's `connections` into the keyword it
    uses, the target it names and the local ports it lists."""
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
    local_ports: list[str] = []
    if ports is None:
        if port is not None:
            assert isinstance(port, str)
            local_ports.append(port)
    elif isinstance(ports, str):
        local_ports.append(ports)
    else:
        local_ports.extend(ports)

    all_keys = [*_SIGNAL_TYPE_TO_CONNECTION_TYPE, T_TO]

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
    if not remote_path:
        raise LabOneQException(
            f"Missing path: specify '{signal_type_keyword}: <group>{qct_path.Separator}<line>'"
        )
    assert isinstance(remote_path, str)
    if signal_type_keyword != T_TO and not _is_valid_remote_path(remote_path):
        raise LabOneQException(
            f"Invalid path: specify '{signal_type_keyword}: <group>{qct_path.Separator}<line>'"
        )
    return signal_type_keyword, remote_path, local_ports


def make_zi_devices(
    instruments: InstrumentsType,
    connections: ConnectionsType,
    server_finder: Callable[[str], str | None],
    setup: DeviceSetup,
):
    # Instruments are parsed in registration order, which is the order they
    # appear in `setup.instruments`.
    for desc_key, model in registry.models_with_descriptor_key():
        make_connection = model.descriptor_connection_decoder or _make_connection
        for uid, address, interface, options in _iterate_over_descriptors_of_type(
            instruments, desc_key
        ):
            conns = connections.get(uid, [])
            # NOTE: The reference clock source is not part of the instrument descriptor, but
            #       part of the connection descriptor. It is therefore parsed here.
            #       Reference clock has no good model as a connection.
            conns, ref_clk = parse_reference_clock(conns)
            dev = model.cls(
                **_skip_nones(
                    uid=uid,
                    server_uid=server_finder(uid),
                    address=address,
                    interface=interface,
                    device_options=options,
                    reference_clock_source=ref_clk,
                )
            )
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
                    msg = f"Error '{uid}' ({signal_type_keyword}, {remote_path}): "
                    msg += str(e)
                    raise LabOneQException(msg) from e


def _make_connection(
    signal_type_kw: str, remote_path: str, local_ports: list[str]
) -> InternalConnection | SignalConnection:
    """Build the connection one decoded descriptor entry denotes."""
    if signal_type_kw == T_TO:
        return InternalConnection(to=remote_path, from_port=local_ports)
    return SignalConnection(
        uid=remote_path,
        type=_SIGNAL_TYPE_TO_CONNECTION_TYPE[signal_type_kw],
        ports=local_ports,
    )


def parse_reference_clock(
    connection: list[dict | str],
) -> tuple[list[dict | str], ReferenceClockSource | None]:
    out = list(connection)
    clk = None
    for d in out:
        if d == T_EXTCLK:
            clk = ReferenceClockSource.EXTERNAL
            out.remove(d)
        elif d == T_INTCLK:
            clk = ReferenceClockSource.INTERNAL
            out.remove(d)
    return out, clk


def make_unmanaged_instrument(
    instruments: InstrumentsType,
    server_finder: Callable[[str], str | None],
    setup: DeviceSetup,
):
    for dev_type, devices in instruments.items():
        if registry.model_by_descriptor_key(dev_type) is None:
            for descriptor in devices:
                uid = descriptor[T_UID]
                dev = _unmanaged_instrument(
                    dev_type,
                    **_skip_nones(
                        server_uid=server_finder(uid),
                        uid=uid,
                        address=descriptor[T_ADDRESS],
                        interface=descriptor.get(T_INTERFACE),
                    ),
                )
                modifier.add_instrument(setup, dev)


def _unmanaged_instrument(device_type: str, **fields) -> Instrument:
    """Build the instrument a device type no registered instrument claims falls
    back to."""
    factory = registry.unmanaged_instrument_factory()
    if factory is None:
        raise LabOneQException(
            f"Instruments of type '{device_type}' are not provided by any "
            f"installed backend."
        )
    return factory(device_type, **fields)


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
    qubits = {}
    for desc in qubit_descriptor:
        try:
            _validate_qubit_descriptor(desc)
        except (TypeError, ValueError) as error:
            msg = error.args[0]
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


def _validate_qubit_descriptor(qubit_descriptor: dict):
    if "name" not in qubit_descriptor:
        raise ValueError("'name' is a required property")

    if "type" not in qubit_descriptor:
        raise ValueError("'type' is a required property")

    if keys := (qubit_descriptor.keys() - {"name", "type"}):
        raise ValueError(f"unexpected keys: {', '.join(sorted(keys))}")

    name = qubit_descriptor["name"]

    if not isinstance(name, (str, list)):
        raise TypeError("'name' is not of type 'string', 'array'")

    name = name if isinstance(name, list) else [name]

    if any(not isinstance(x, str) for x in name):
        raise TypeError("'name' contains element that is not of type 'string'")

    if len(name) != len(set(name)):
        raise ValueError(f"{name} has non-unique elements")

    qubit_type = qubit_descriptor["type"]
    if not isinstance(qubit_type, str):
        raise TypeError("'type' is not of type 'string'")


class _DeviceSetupGenerator:
    @staticmethod
    def from_descriptor(
        yaml_text: str,
        server_host: str | None = None,
        server_port: str | int | None = None,
        setup_name: str | None = None,
    ):
        setup_desc = safe_load(yaml_text)

        return _DeviceSetupGenerator.from_dicts(
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
        server_port: str | int | None = None,
        setup_name: str | None = None,
    ):
        with open(filepath) as fp:
            setup_desc = safe_load(fp)

        return _DeviceSetupGenerator.from_dicts(
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
        instruments: InstrumentsType | None = None,
        connections: ConnectionsType | None = None,
        dataservers: DataServersType | None = None,
        qubits: list | None = None,
        server_host: str | None = None,
        server_port: str | int | None = None,
        setup_name: str | None = None,
    ):
        # To avoid circular imports
        from laboneq.dsl import quantum
        from laboneq.dsl.device.device_setup import DeviceSetup

        if qubits is None:
            qubits = []

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
                    "port": "8004" if server_port is None else str(server_port),
                }
            }

        if dataservers is None:
            dataservers = {}

        def _make_server_entry(
            server_uid: str, server_def: dict[str, str | list[str]]
        ) -> tuple[DataServer, list[str]]:
            host = server_def.get("host")
            port = server_def.get("port")
            instruments = server_def.get("instruments")
            # server_def might incorrectly contain None values
            if host is None:
                host = ""
            if port is None:
                port = 8004
            if instruments is None:
                instruments = []
            assert isinstance(host, str)
            assert isinstance(port, (str, int))
            assert isinstance(instruments, list)
            return DataServer(
                uid=server_uid, host=host, port=port, api_level=6
            ), instruments

        # Construct servers
        servers: list[tuple[DataServer, list[str]]] = [
            _make_server_entry(server_uid, server_def)
            for server_uid, server_def in dataservers.items()
        ]

        def server_finder(device_uid: str) -> str | None:
            if not servers:
                return None
            default_data_server: DataServer | None = None
            explicit_data_server: DataServer | None = None
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

    @staticmethod
    def from_instrument_topology(
        instrument_topology: InstrumentTopology,
        uid: str | None = None,
    ) -> DeviceSetup:
        # To avoid circular imports
        from laboneq.dsl.device.device_setup import DeviceSetup

        setup = DeviceSetup(
            **_skip_nones(
                uid=uid,
                setup_description=instrument_topology.setup_description,
            )
        )
        # Assigned rather than passed to the constructor: dispatching to a
        # controller service is a transport detail, not part of the setup's
        # public description.
        setup.controller_service_url = instrument_topology.controller_service_url

        if instrument_topology.servers:
            for server in instrument_topology.servers:
                setup.add_dataserver(
                    host=server.host if server.host is not None else "",
                    port=server.port if server.port is not None else 8004,
                    uid=server.uid,
                )

        for instrument in instrument_topology.instruments:
            options = "/".join(instrument.options) if instrument.options else None
            model = registry.model_by_descriptor_key(instrument.device_type)
            if model is not None:
                dev = model.cls(
                    uid=instrument.uid,
                    address=instrument.address,
                    **_skip_nones(
                        server_uid=instrument.server_uid,
                        interface=instrument.interface,
                        device_options=options,
                        reference_clock_source=instrument.reference_clock_source,
                    ),
                )
            else:
                dev = _unmanaged_instrument(
                    instrument.device_type,
                    uid=instrument.uid,
                    address=instrument.address,
                    **_skip_nones(
                        server_uid=instrument.server_uid,
                        interface=instrument.interface,
                        device_options=options,
                        reference_clock_source=instrument.reference_clock_source,
                    ),
                )
            setup.add_instruments(dev)

        connections_by_instrument: dict[
            str, list[SignalConnection | InternalConnection]
        ] = {}
        connection: SignalConnection | InternalConnection
        for entry in instrument_topology.connections:
            if entry.signal_type is not None:
                connection_type = _SIGNAL_TYPE_TO_CONNECTION_TYPE.get(entry.signal_type)
                if connection_type is None:
                    raise LabOneQException(
                        f"Instrument '{entry.instrument_uid}': unrecognized "
                        f"signal_type '{entry.signal_type}'."
                    )
                ports = [entry.primary_port]
                if entry.secondary_port is not None:
                    ports.append(entry.secondary_port)
                connection = SignalConnection(
                    uid=entry.signal_name, type=connection_type, ports=ports
                )
            else:
                connection = InternalConnection(
                    to=entry.primary_port, from_port=entry.secondary_port
                )
            connections_by_instrument.setdefault(entry.instrument_uid, []).append(
                connection
            )

        for instrument_uid, connections in connections_by_instrument.items():
            setup.add_connections(instrument_uid, *connections)

        return setup
