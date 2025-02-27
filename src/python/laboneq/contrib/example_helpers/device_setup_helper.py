# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time

from zhinst.toolkit.session import Session as TKSession

from laboneq.core.exceptions import LabOneQException


def create_TKSession(
    dataserver_ip: str = "localhost",
    server_port: int | None = None,
    hub_id: str | None = None,
    device_ids: list[str] | None = None,
):
    """
    Args:
        dataserver_ip: IP address of the dataserver to use as a string.
            Defaults to "localhost".
        hub_id: Identifier of the PQSC or QHUB instrument to use to coordinate the setup, e.g. "DEV10001".
            Defaults to None.
        device_ids: List of Instrument identifiers as strings, i.e. ["DEV1234", "DEV2345"].
            Defaults to None.
    """
    # create session
    my_session = TKSession(dataserver_ip, server_port=server_port)
    # connect pqsc / QHUB
    if hub_id is not None:
        my_session.connect_device(hub_id)
    # connect instruments
    if device_ids is not None:
        for id in device_ids:
            my_session.connect_device(id)

    return my_session


def get_single_instrument_options(
    session: TKSession,
    device_id: str,
):
    """Return the options from a single instrument connected to the toolkit session.
    Args:
        session: A connected toolkit session.
        device_id: Identifier of an instrument connected to the toolkit session.
    """
    connected_devices = [id.upper() for id in session.devices]
    if device_id.upper() in connected_devices:
        device = session.devices[device_id.upper()]
        return device.device_type + "/" + "/".join(device.device_options.split())
    else:
        raise LabOneQException(f"Instrument {device_id} not connected to session.")


def return_instrument_options(
    session: TKSession,
    device_ids: list[str],
):
    """Return all options for a list of instruments connected to the toolkit session.
    Args:
        session: A connected toolkit session.
        device_ids: List of Instrument identifiers as strings, i.e. ["DEV1234", "DEV2345"].
    """
    options = [get_single_instrument_options(session, id) for id in device_ids]

    return dict(zip(device_ids, options))


def connect_zsync(
    session: TKSession,
    device_ids: list[str],
    waiting_time: float = 2,
):
    """Check if instruments are already connected to zsync and establish connection if not.
    Args:
        session: A connected toolkit session.
        device_ids: List of Instrument identifiers as strings, i.e. ["DEV1234", "DEV2345"].
        waiting_time: Waiting time in seconds after setting the nodes to ensure connection is established.
            Defaults to 2 seconds.
    """
    for instrument in device_ids:
        my_instrument = session.devices[instrument]
        # check clock source
        if "HDAWG" in my_instrument.device_type:
            clock_source = my_instrument.system.clocks.referenceclock.source
        elif "SHF" in my_instrument.device_type:
            clock_source = my_instrument.system.clocks.referenceclock.in_.source
        else:
            pass
        # check if already connected, if not, initiate connection
        if not clock_source().value == 2:
            clock_source(2)
    time.sleep(waiting_time)


def return_instrument_zsync_ports(
    session: TKSession,
    hub_id: str,
    device_ids: list[str],
):
    """Return the zsync ports for a list of instruments connected to a toolkit session.
    Args:
        session: A connected toolkit session.
        hub_id: Identifier of the PQSC or QHUB instrument to use to coordinate the setup, e.g. "DEV10001".
        device_ids: List of Instrument identifiers as strings, i.e. ["DEV1234", "DEV2345"].
    """
    connected_devices = [id.upper() for id in session.devices]
    if hub_id.upper() in connected_devices:
        hub = session.devices[hub_id.upper()]
    else:
        raise LabOneQException(f"Instrument {hub_id} not connected to session.")
    zsync_dict = hub.zsyncs["*"].connection.serial()
    zsync_ports = [
        list(zsync_dict.keys())[list(zsync_dict.values()).index(id[3:])].split("/")[-3]
        for id in device_ids
    ]
    return dict(zip(device_ids, zsync_ports))
