# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate a device setup and a list of qubits from parameters, to use with the tutorials and how_to notebooks in the LabOne Q repository"""

from __future__ import annotations

from laboneq.contrib.example_helpers.device_setup_helper import (
    create_TKSession,
    connect_zsync,
    return_instrument_options,
    return_instrument_zsync_ports,
)
from laboneq.contrib.example_helpers.example_notebook_helper import (
    create_dummy_transmon,
    generate_dummy_transmon_parameters,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import HDAWG, PQSC, QHUB, SHFQA, SHFQC, SHFSG, UHFQA


# generate a device setup from a list of Instrument objects
def generate_device_setup(
    number_qubits: int = 6,
    qhub: list[dict[str]] | None = None,
    pqsc: list[dict[str]] | None = None,
    hdawg: list[dict[str]] | None = None,
    shfsg: list[dict[str]] | None = None,
    shfqc: list[dict[str]] | None = None,
    shfqa: list[dict[str]] | None = None,
    uhfqa: list[dict[str]] | None = None,
    multiplex_drive_lines: bool = False,
    include_flux_lines: bool = False,
    drive_only: bool = False,
    readout_only: bool = False,
    server_host: str = "localhost",
    server_port: str = "8004",
    setup_name: str = "my_QCCS",
    query_zsync: bool = False,
    query_options: bool = False,
    debug: bool = False,
) -> DeviceSetup:
    """A function to generate a DeviceSetup given a list of devices, based on standardised wiring assumptions.

    With this function, you can auto-generate a device setup based on a list of input parameters,
    making some standard assumptions about wiring configuration.

    Args:
        number_qubits: The number of qubits that the device setup shall be configured for.
            Defaults to 6.
        qhub: The device id and additional properties of your QHUB as a list of dictionaries
            (e.g. `[{"serial": "DEV10XX0", "external_clock": False, "usb": False}]`).
            Note: only one QHUB or PQSC is possible per set-up.
        pqsc: The device id and additional properties of your PQSC as a list of dictionaries
            (e.g. `[{"serial": "DEV10XX0", "external_clock": False, "usb": False}]`).
            Note: only one PQSC or QHUB is possible per set-up.
        hdawg: The device id(s) and additional properties  of your HDAWG instruments as a list of dictionaries
            (e.g.`[{"serial": "DEV8XXX", "usb": False, "zsync": 2, "dio": None, "options": None, "number_of_channels": 8}]`).
        uhfqa: The device id(s) and additional properties of your UHFQA instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV2XXX", "usb": False, "readout_multiplex": 6}]`).
            Note: UHFQA instruments cannot be used in combination with SHF instruments.
        shfsg: The device id(s) and additional properties of your SHFSG instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV12XXX", "usb": False, "number_of_channels": 8, "options": None, "zsync": 3}]`).
        shfqc: The device id(s) and additional properties of your SHFQC instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV12XXX", "usb": False, "number_of_channels": 6, "readout_multiplex": 6, "options": None, "zsync": 4}]`).
        shfqa: The device id(s) and additional properties of your SHFQA instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV12XXX", "usb": False, "number_of_channels": 4, "readout_multiplex": 6, "options": None, "zsync": 5}]`).
        multiplex_drive_lines: Whether to add logical signals that are mutliplexed t othe drive lines,
            e.g. for cross-resonance or e-f driving.
            Defaults to False.
        include_flux_lines: Whether to include flux lines in the setup.
            Defaults to False.
        drive_only: Whether to generates a device setup without readout or acquisition lines.
            Defaults to False.
        readout_only: Whether to generates a device setup without any drive or flux lines.
            Defaults to False.
        server_host: The IP address of the LabOne dataserver used to connect to the instruments.
            Defaults to "localhost".
        server_port: The port number of the LabOne dataserver used to connect to the instruments.
            Defaults to the LabOne default setting "8004".
        setup_name: The name of your setup.
            Defaults to "my_QCCS"
        query_zsync: Whether to query the PQSC or QHUB for the zsync ports of all connected instruments.
            Defaults to False.
        query_options: Whether to query all connected instruments for their options.
            Defaults to False.
        debug: Whether to print optional debug information to console.
            Defaults to False.

    Returns:
        A LabOne Q DeviceSetup object with the specified configuration.
    """

    # debug printout helper
    def print_debug(*args, **kwargs):
        if debug:
            print(*args, **kwargs)

    # redefine inputs for simpler conditionals later
    uhfqa = uhfqa or []
    hdawg = hdawg or []
    shfqa = shfqa or []
    shfsg = shfsg or []
    shfqc = shfqc or []
    pqsc = pqsc or []
    qhub = qhub or []

    # check input for consistency
    if uhfqa and (shfsg or shfqa or shfqc):
        raise LabOneQException(
            "Device Setup generation failed: Mixing SHF and UHF instruments in a single setup is not supported."
        )
    gen1_setup = False
    if not pqsc or qhub:
        if uhfqa and hdawg:
            if len(uhfqa) + len(hdawg) > 2:
                raise LabOneQException(
                    "Device Setup generation failed: Without PQSC, only a single UHFQA can be combined together with a single HDAWG."
                )
            else:
                print_debug("INFO: Small setup detected - HDAWG + UHFQA.")
                gen1_setup = True
        if hdawg and not (uhfqa or shfsg or shfqa or shfqc):
            print_debug("INFO: Small setup detected - HDAWG only.")
            gen1_setup = True
            drive_only = True
        if shfsg and not (hdawg or shfqa or shfqc):
            print_debug("INFO: Small setup detected - SHFSG only.")
            drive_only = True
        if shfqc and not (hdawg or shfqa or shfsg):
            print_debug("INFO: Small setup detected - SHFQC only.")
        if shfqa and not (hdawg or shfqc or shfsg):
            print_debug("INFO: Small setup detected - SHFQA only.")
    elif len(pqsc) + len(qhub) > 1:
        raise LabOneQException(
            "Device Setup generation failed: Only a single PQSC or QHub is supported in a QCCS setup."
        )
    elif uhfqa and hdawg:
        gen1_setup = True
    if gen1_setup:
        if include_flux_lines > 0:
            raise LabOneQException(
                "Device Setup generation failed: Flux lines are not supported for Gen1 setups."
            )
        elif multiplex_drive_lines:
            raise LabOneQException(
                "Device Setup generation failed: Cross-resonance drive lines are not supported for Gen1 setups."
            )
    if gen1_setup and qhub:
        raise LabOneQException(
            "Device Setup generation failed: QHUB does not support Gen1 setups."
        )
    if qhub and pqsc:
        raise LabOneQException(
            "Device Setup generation failed: Setups can only contain either a single PQSC or a single QHUB, not both."
        )

    # query instruments for options / zsync connections
    # TODO: what about UHFQA?
    if query_options or query_zsync:
        all_instruments = [hdawg, shfsg, shfqa, shfqc]
        # generate lists of instrument ids
        device_ids = [instrument["serial"] for instrument in hdawg]
        device_ids.extend(instrument["serial"] for instrument in shfsg)
        device_ids.extend(instrument["serial"] for instrument in shfqa)
        device_ids.extend(instrument["serial"] for instrument in shfqc)
        hub_id = pqsc[0]["serial"] if pqsc else qhub[0]["serial"]
        # create connected toolkit session
        tk_session = create_TKSession(
            dataserver_ip=server_host,
            server_port=int(server_port),
            hub_id=hub_id,
            device_ids=device_ids,
        )
        # query and update instrument options
        if query_options:
            option_dict = return_instrument_options(
                session=tk_session, device_ids=device_ids
            )
            for serial, option in option_dict.items():
                for instrument_list in all_instruments:
                    for instrument in instrument_list:
                        if instrument["serial"] == serial:
                            instrument["options"] = option
                            break
        # query and update zsync port connections
        if query_zsync:
            connect_zsync(session=tk_session, device_ids=device_ids, waiting_time=2)
            zsync_dict = return_instrument_zsync_ports(
                session=tk_session, hub_id=hub_id, device_ids=device_ids
            )
            for serial, port in zsync_dict.items():
                for instrument_list in all_instruments:
                    for instrument in instrument_list:
                        if instrument["serial"] == serial:
                            instrument["zsync"] = port
                            break

    # generate device setup including dataserver configuration
    device_setup = DeviceSetup(uid=setup_name)
    device_setup.add_dataserver(host=server_host, port=server_port)

    # add instruments, collect information on setup capabilities from instrument lists
    instrument_list = {}
    drive_lines = []
    number_drive_lines = 0
    flux_lines = []
    number_flux_lines = 0
    readout_acquire_lines = []
    number_readout_acquire_lines = 0
    zsync = {}
    dio = {}

    # UHFQA
    for id, instrument in enumerate(uhfqa):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for UHFQA instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "readout_multiplex" not in instrument:
            instrument["readout_multiplex"] = 6
        device_setup.add_instruments(
            UHFQA(
                uid=f"uhfqa_{id}",
                address=instrument["serial"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        actual_multiplex = min(10, instrument["readout_multiplex"])
        readout_acquire_lines.extend(
            [
                {
                    "device": f"uhfqa_{id}",
                    "port_out": ["SIGOUTS/0", "SIGOUTS/1"],
                    "port_in": None,
                    "multiplex": actual_multiplex,
                }
            ]
        )
        number_readout_acquire_lines += actual_multiplex
        instrument_list[instrument["serial"]] = f"uhfqa_{id}"

    # HDAWG
    for id, instrument in enumerate(hdawg):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for HDAWG instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "options" not in instrument:
            instrument["options"] = None
        if "zsync" not in instrument:
            instrument["zsync"] = None
        if "dio" not in instrument:
            instrument["dio"] = None
        if "number_of_channels" not in instrument:
            instrument["number_of_channels"] = 8
        device_setup.add_instruments(
            HDAWG(
                uid=f"hdawg_{id}",
                address=instrument["serial"],
                device_options=instrument["options"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        if gen1_setup:
            drive_lines.extend(
                [
                    {
                        "device": f"hdawg_{id}",
                        "port": [f"SIGOUTS/{2 * it}", f"SIGOUTS/{2 * it + 1}"],
                    }
                    for it in range(int(instrument["number_of_channels"] / 2))
                ]
            )
            number_drive_lines += instrument["number_of_channels"] / 2
        else:
            flux_lines.extend(
                [
                    {
                        "device": f"hdawg_{id}",
                        "port": f"SIGOUTS/{it}",
                    }
                    for it in range(instrument["number_of_channels"])
                ]
            )
            number_flux_lines += instrument["number_of_channels"]
        instrument_list[instrument["serial"]] = f"hdawg_{id}"
        if instrument["zsync"] is not None:
            zsync[instrument["serial"]] = instrument["zsync"]
        if instrument["dio"] is not None:
            dio[instrument["serial"]] = instrument["dio"]

    # check if combination and wiring of HDAWG and UFQA instruments works
    # TODO: ensure that for each UHFQA there is exactly one HDAWG connected to it via DIO
    if uhfqa:
        if hdawg:
            gen1_setup = True
            if not dio:
                raise LabOneQException(
                    "Device Setup generation failed: DIO connection information between UHFQA and HDAWG instruments is required."
                )
        else:
            raise LabOneQException(
                "Device Setup generation failed: UHFQA requires a HDAWG to be used with LabOne Q."
            )

    # SHFSG
    for id, instrument in enumerate(shfsg):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for SHFSG instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "options" not in instrument:
            instrument["options"] = None
        if "zsync" not in instrument:
            instrument["zsync"] = None
        if "number_of_channels" not in instrument:
            instrument["number_of_channels"] = 8
        device_setup.add_instruments(
            SHFSG(
                uid=f"shfsg_{id}",
                address=instrument["serial"],
                device_options=instrument["options"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        drive_lines.extend(
            [
                {"device": f"shfsg_{id}", "port": f"SGCHANNELS/{it}/OUTPUT"}
                for it in range(instrument["number_of_channels"])
            ]
        )
        number_drive_lines += instrument["number_of_channels"]
        instrument_list[instrument["serial"]] = f"shfsg_{id}"
        if instrument["zsync"] is not None:
            zsync[instrument["serial"]] = instrument["zsync"]

    # SHFQA
    for id, instrument in enumerate(shfqa):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for SHFQA instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "options" not in instrument:
            instrument["options"] = None
        if "zsync" not in instrument:
            instrument["zsync"] = None
        if "readout_multiplex" not in instrument:
            instrument["readout_multiplex"] = 6
        if "number_of_channels" not in instrument:
            instrument["number_of_channels"] = 4
        device_setup.add_instruments(
            SHFQA(
                uid=f"shfqa_{id}",
                address=instrument["serial"],
                device_options=instrument["options"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        actual_multiplex = min(16, instrument["readout_multiplex"])
        readout_acquire_lines.extend(
            [
                {
                    "device": f"shfqa_{id}",
                    "port_out": f"QACHANNELS/{it}/OUTPUT",
                    "port_in": f"QACHANNELS/{it}/INPUT",
                    "multiplex": actual_multiplex,
                }
                for it in range(instrument["number_of_channels"])
            ]
        )
        number_readout_acquire_lines += (
            instrument["number_of_channels"] * actual_multiplex
        )
        instrument_list[instrument["serial"]] = f"shfqa_{id}"
        if instrument["zsync"] is not None:
            zsync[instrument["serial"]] = instrument["zsync"]

    # SHFQC
    for id, instrument in enumerate(shfqc):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for SHFQC instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "options" not in instrument:
            instrument["options"] = None
        if "zsync" not in instrument:
            instrument["zsync"] = None
        if "readout_multiplex" not in instrument:
            instrument["readout_multiplex"] = 6
        if "number_of_channels" not in instrument:
            instrument["number_of_channels"] = 6
        device_setup.add_instruments(
            SHFQC(
                uid=f"shfqc_{id}",
                address=instrument["serial"],
                device_options=instrument["options"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        drive_lines.extend(
            [
                {"device": f"shfqc_{id}", "port": f"SGCHANNELS/{it}/OUTPUT"}
                for it in range(instrument["number_of_channels"])
            ]
        )
        number_drive_lines += instrument["number_of_channels"]
        # drive_instruments[f"shfqc_{id}"] = {}
        actual_multiplex = min(16, instrument["readout_multiplex"])
        readout_acquire_lines.extend(
            [
                {
                    "device": f"shfqc_{id}",
                    "port_out": "QACHANNELS/0/OUTPUT",
                    "port_in": "QACHANNELS/0/INPUT",
                    "multiplex": actual_multiplex,
                }
            ]
        )
        number_readout_acquire_lines += actual_multiplex
        instrument_list[instrument["serial"]] = f"shfqc_{id}"
        if instrument["zsync"] is not None:
            zsync[instrument["serial"]] = instrument["zsync"]

    # PQSC
    for id, instrument in enumerate(pqsc):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for PQSC instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "external_clock" not in instrument:
            instrument["external_clock"] = False
        device_setup.add_instruments(
            PQSC(
                uid=f"pqsc_{id}",
                address=instrument["serial"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        instrument_list[instrument["serial"]] = f"pqsc_{id}"

    # QHub
    for id, instrument in enumerate(qhub):
        if "serial" not in instrument:
            raise LabOneQException(
                f"Device Setup generation failed: Serial not provided for QHUB instrument - {id} - {instrument}."
            )
        if "usb" not in instrument:
            instrument["usb"] = False
        if "external_clock" not in instrument:
            instrument["external_clock"] = False
        device_setup.add_instruments(
            QHUB(
                uid=f"qhub_{id}",
                address=instrument["serial"],
                interface="1GBe" if not instrument["usb"] else "usb",
            )
        )
        instrument_list[instrument["serial"]] = f"pqsc_{id}"

    # check that instruments supplied are sufficient for specified needs
    if number_readout_acquire_lines < number_qubits and not drive_only:
        raise LabOneQException(
            f"Device Setup generation failed: not enought readout / acquire lines configurable ({number_readout_acquire_lines}) for the specified number of qubits ({number_qubits})"
        )
    if number_drive_lines < number_qubits:
        raise LabOneQException(
            f"Device Setup generation failed: not enought drive lines configurable ({number_drive_lines}) for the specified number of qubits ({number_qubits})"
        )
    if include_flux_lines and (number_flux_lines < number_qubits):
        raise LabOneQException(
            f"Device Setup generation failed: not enought flux lines configurable ({number_flux_lines}) for the specified number of qubits ({number_qubits})"
        )

    # add logical signal lines for the specified number of qubits
    qubits = [f"q{it}" for it in range(number_qubits)]

    # measurement lines
    if not drive_only:
        # add readout and acquire lines
        readout_index = 0
        current_readout = readout_acquire_lines[readout_index]
        readout_multiplex = current_readout["multiplex"]
        current_multiplex = 1
        for qubit in qubits:
            # advance in list of readout lines
            if current_multiplex > readout_multiplex:
                readout_index += 1
                current_readout = readout_acquire_lines[readout_index]
                readout_multiplex = current_readout["multiplex"]
                current_multiplex = 1
            device_setup.add_connections(
                current_readout["device"],
                create_connection(
                    to_signal=f"{qubit}/measure", ports=current_readout["port_out"]
                ),
                create_connection(
                    to_signal=f"{qubit}/acquire", ports=current_readout["port_in"]
                ),
            )
            current_multiplex += 1

    # control lines
    if not readout_only:
        # add drive and cr lines
        drive_index = 0
        for qubit in qubits:
            current_drive = drive_lines[drive_index]
            # advance in list of drive lines
            device_setup.add_connections(
                current_drive["device"],
                create_connection(
                    to_signal=f"{qubit}/drive", ports=current_drive["port"]
                ),
            )
            # add drive lines for higher qubit states - multiplexed with g-e drive lines
            if multiplex_drive_lines:
                device_setup.add_connections(
                    current_drive["device"],
                    create_connection(
                        to_signal=f"{qubit}/drive_ef", ports=current_drive["port"]
                    ),
                )
            drive_index += 1

        # add flux lines
        if include_flux_lines:
            flux_index = 0
            for qubit in qubits:
                current_flux = flux_lines[flux_index]
                # advance in list of flux lines
                device_setup.add_connections(
                    current_flux["device"],
                    create_connection(
                        to_signal=f"{qubit}/flux", ports=current_flux["port"]
                    ),
                )
                flux_index += 1

    # add DIO connections
    if dio and gen1_setup:
        for dio_in, dio_out in dio.items():
            # check if uhfqa is present in instrument list
            if dio_out in instrument_list:
                device_setup.add_connections(
                    instrument_list[dio_in],
                    create_connection(
                        to_instrument=instrument_list[dio_out], ports="DIOS/0"
                    ),
                )
            elif not drive_only:
                raise LabOneQException(
                    f"Device Setup generation failed: UHFQA {dio_out} is not part of device setup "
                )
    # add ZSync connections
    if zsync:
        if pqsc:
            for zsync_in, zsync_id in zsync.items():
                # ensure that the instrument is already added to the device setup
                if zsync_in in instrument_list.keys():
                    device_setup.add_connections(
                        "pqsc_0",
                        create_connection(
                            to_instrument=instrument_list[zsync_in],
                            ports=f"ZSYNCS/{zsync_id}",
                        ),
                    )
        elif qhub:
            for zsync_in, zsync_id in zsync.items():
                # ensure that the instrument is already added to the device setup
                if zsync_in in instrument_list.keys():
                    device_setup.add_connections(
                        "qhub_0",
                        create_connection(
                            to_instrument=instrument_list[zsync_in],
                            ports=f"ZSYNCS/{zsync_id}",
                        ),
                    )

    return device_setup


def generate_device_setup_qubits(
    number_qubits: int = 6,
    qhub: list[dict[str]] | None = None,
    pqsc: list[dict[str]] | None = None,
    hdawg: list[dict[str]] | None = None,
    shfsg: list[dict[str]] | None = None,
    shfqc: list[dict[str]] | None = None,
    shfqa: list[dict[str]] | None = None,
    uhfqa: list[dict[str]] | None = None,
    multiplex_drive_lines: bool = False,
    include_flux_lines: bool = False,
    drive_only: bool = False,
    readout_only: bool = False,
    server_host: str = "localhost",
    server_port: str = "8004",
    setup_name: str = "my_QCCS",
    include_qubits: bool = True,
    calibrate_setup: bool = True,
    query_zsync: bool = False,
    query_options: bool = False,
    debug: bool = False,
):
    """A function to generate a DeviceSetup and a list of Transmon qubits
    given a list of devices, based on standardised wiring assumptions.

    With this function, you can auto-generate a device setup and a list of Transmon qubits
    based on a list of input parameters, making some standard assumptions about wiring configuration.

    Args:
        number_qubits: The number of qubits that the device setup shall be configured for.
        qhub: The device id and additional properties of your QHUB as a list of dictionaries
            (e.g. `[{"serial": "DEV10XX0", "external_clock": False, "usb": False}]`).
            Note: only one QHUB or PQSC is possible per set-up.
        pqsc: The device id and additional properties of your PQSC as a list of dictionaries
            (e.g. `[{"serial": "DEV10XX0", "external_clock": False, "usb": False}]`).
            Note: only one PQSC or QHUB is possible per set-up.
        hdawg: The device id(s) and additional properties  of your HDAWG instruments as a list of dictionaries
            (e.g.`[{"serial": "DEV8XXX", "usb": False, "zsync": 2, "dio": None, "options": None, "number_of_channels": 8}]`).
        uhfqa: The device id(s) and additional properties of your UHFQA instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV2XXX", "usb": False, "readout_multiplex": 6}]`).
            Note: UHFQA instruments cannot be used in combination with SHF instruments.
        shfsg: The device id(s) and additional properties of your SHFSG instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV12XXX", "usb": False, "number_of_channels": 8, "options": None, "zsync": 3}]`).
        shfqc: The device id(s) and additional properties of your SHFQC instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV12XXX", "usb": False, "number_of_channels": 6, "readout_multiplex": 6, "options": None, "zsync": 4}]`).
        shfqa: The device id(s) and additional properties of your SHFQA instruments as a list of dictionaries
            (e.g. `[{"serial": "DEV12XXX", "usb": False, "number_of_channels": 4, "readout_multiplex": 6, "options": None, "zsync": 5}]`).
        multiplex_drive_lines: Whether to add logical signals that are mutliplexed t othe drive lines,
            e.g. for cross-resonance or e-f driving.
            Defaults to False.
        include_flux_lines: Whether to include flux lines in the setup.
            Defaults to False.
        drive_only: Whether to generates a device setup without readout or acquisition lines.
            Defaults to False.
        readout_only: Whether to generates a device setup without any drive or flux lines.
            Defaults to False.
        server_host: The IP address of the LabOne dataserver used to connect to the instruments.
            Defaults to "localhost".
        server_port: The port number of the LabOne dataserver used to connect to the instruments.
            Defaults to the LabOne default setting "8004".
        setup_name: The name of your setup. Defaults to "my_QCCS".
        include_qubits: Whether to include the qbits in the device setup itself, under the `DeviceSetup.qubits` property,
            in addition to returning them.
            Defaults to True.
        calibrate_setup: Whether to use the qubit properties to calibrate the device setup.
            Defaults to True.
        query_zsync: Whether to query the PQSC or QHUB for the zsync ports of all connected instruments.
            Defaults to False.
        query_options: Whether to query all connected instruments for their options.
            Defaults to False.
        debug: Whether to print optional debug information to console.
            Defaults to False.

    Returns:
        A LabOne Q DeviceSetup object with the specified configuration
            as well as a list of Transmon qubits configured for the given DeviceSetup.
    """

    device_setup = generate_device_setup(
        number_qubits=number_qubits,
        qhub=qhub,
        pqsc=pqsc,
        hdawg=hdawg,
        shfsg=shfsg,
        shfqc=shfqc,
        shfqa=shfqa,
        uhfqa=uhfqa,
        multiplex_drive_lines=multiplex_drive_lines,
        include_flux_lines=include_flux_lines,
        drive_only=drive_only,
        readout_only=readout_only,
        server_host=server_host,
        server_port=server_port,
        setup_name=setup_name,
        query_zsync=query_zsync,
        query_options=query_options,
        debug=debug,
    )

    dummy_qubit_parameters = generate_dummy_transmon_parameters(
        number_of_qubits=number_qubits
    )

    qubits = [
        create_dummy_transmon(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(number_qubits)
    ]
    if include_qubits:
        device_setup.qubits = qubits
    if calibrate_setup:
        for qubit in qubits:
            device_setup.set_calibration(qubit.calibration())

    return device_setup, qubits
