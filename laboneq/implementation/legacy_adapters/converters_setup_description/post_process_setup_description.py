# Copyright 2023 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from laboneq.data.setup_description import (
    Connection,
    DeviceType,
    LogicalSignalGroup,
    PhysicalChannel,
    PhysicalChannelType,
    Port,
    Setup,
    SetupInternalConnection,
)
from laboneq.dsl.device.instrument import Instrument as InstrumentDSL
from laboneq.dsl.device.io_units.physical_channel import (
    PhysicalChannelType as PhysicalChannelTypeDSL,
)

in_postprocess = False


def post_process(source, target, conversion_function_lookup):
    global in_postprocess
    if in_postprocess:
        return target

    device_types = {e.name: e for e in DeviceType}
    if type(source).__name__ in device_types:
        in_postprocess = True
        retval = conversion_function_lookup(InstrumentDSL)(source)
        in_postprocess = False
        retval.address = source.address
        retval.device_type = device_types[type(source).__name__]
        retval.server = source.server_uid
        return retval

    if type(target) == Setup:
        post_process_setup(source, target)

    if type(target) == LogicalSignalGroup:
        target.logical_signals = {
            ls.uid.split("/")[1]: ls for ls in target.logical_signals
        }
    return target


def is_node_path_in_physical_channel(node_path, physical_channel):
    path_split = node_path.split("/")
    first_part_of_node_path = path_split[0]
    first_path_of_pc_name = physical_channel.name.split("_")[0]
    if not first_part_of_node_path.lower() == first_path_of_pc_name:
        return False
    if len(path_split) >= 3:
        last_parth_of_node_path = path_split[-1]
        last_part_of_pc_name = physical_channel.name.split("_")[-1]
        if not last_parth_of_node_path.lower() == last_part_of_pc_name:
            return False

    channel_numbers_in_pc_name = [
        int(s) for s in physical_channel.name.split("_") if s.isdecimal()
    ]
    if len(channel_numbers_in_pc_name) == 0:
        return False
    channel_numbers_in_node_path = [
        int(s) for s in node_path.split("/") if s.isdecimal()
    ]
    if len(channel_numbers_in_node_path) > 1:
        return False
    if len(channel_numbers_in_node_path) == 0:
        return False
    if channel_numbers_in_node_path[0] not in channel_numbers_in_pc_name:
        return False
    return True


def post_process_setup(dsl_setup, data_setup):
    data_instrument_map = {i.uid: i for i in data_setup.instruments}
    dsl_instrument_map = {i.uid: i for i in dsl_setup.instruments}
    all_pcs = {}
    for device_id, pcg in dsl_setup.physical_channel_groups.items():
        for pc_uid, pc in pcg.channels.items():
            all_pcs[(device_id, pc.name)] = pc

    all_ls = {}
    for lsg in data_setup.logical_signal_groups:
        for ls in lsg.logical_signals.values():
            all_ls[ls.path] = ls

    for i in data_setup.instruments:
        server_uid = i.server
        if server_uid is not None:
            i.server = next(s for s in data_setup.servers if s.uid == server_uid)

        i.physical_channels = [
            PhysicalChannel(
                uid=pc.name,
                type=PhysicalChannelType.IQ_CHANNEL
                if pc.type == PhysicalChannelTypeDSL.IQ_CHANNEL
                else PhysicalChannelType.RF_CHANNEL,
            )
            for k, pc in all_pcs.items()
            if k[0] == i.uid
        ]

        i.ports = []
        i.connections = []
        pcs_of_instrument = [pc for k, pc in all_pcs.items() if k[0] == i.uid]

        for c in dsl_instrument_map[i.uid].connections:
            node_path = c.local_port

            pc_of_connection = next(
                (
                    pc
                    for pc in pcs_of_instrument
                    if is_node_path_in_physical_channel(node_path, pc)
                ),
                None,
            )

            if pc_of_connection is not None:
                pc_of_connection = next(
                    (
                        pc
                        for pc in i.physical_channels
                        if pc.uid == pc_of_connection.name
                    ),
                    None,
                )

            current_port = Port(path=c.local_port, physical_channel=pc_of_connection)
            i.ports.append(current_port)

            if c.remote_path in all_ls and pc_of_connection is not None:
                i.connections.append(
                    Connection(
                        physical_channel=pc_of_connection,
                        logical_signal=all_ls[c.remote_path],
                    )
                )
            elif c.remote_path in data_instrument_map:
                data_setup.setup_internal_connections.append(
                    SetupInternalConnection(
                        from_instrument=i,
                        to_instrument=data_instrument_map[c.remote_path],
                        from_port=current_port,
                    )
                )

    data_setup.servers = {s.uid: s for s in data_setup.servers}
    data_setup.logical_signal_groups = {
        lsg.uid: lsg for lsg in data_setup.logical_signal_groups
    }
