# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate a device setup and a list of qubits from parameters, to use with the tutorials and how_to notebooks in the LabOne Q repository
"""

from __future__ import annotations

from laboneq.contrib.example_helpers.example_notebook_helper import (
    create_dummy_transmon,
    generate_dummy_transmon_parameters,
)
from laboneq.core.exceptions import LabOneQException
from laboneq.dsl.device import DeviceSetup, create_connection
from laboneq.dsl.device.instruments import (
    HDAWG,
    PQSC,
    SHFQA,
    SHFQC,
    SHFSG,
    UHFQA,
)


# generate a device setup from a list of Instrument objects
def generate_device_setup(
    number_qubits: int = 6,
    pqsc: list[str] | None = None,
    hdawg: list[str] | None = None,
    shfsg: list[str] | None = None,
    shfqc: list[str] | None = None,
    shfqa: list[str] | None = None,
    uhfqa: list[str] | None = None,
    number_multiplex: int | None = 6,
    multiplex_drive_lines: bool = False,
    include_flux_lines: bool = False,
    drive_only: bool = False,
    dio: dict[str] | None = None,
    zsync: dict[str | int] | None = None,
    server_host: str = "localhost",
    server_port: str = "8004",
    setup_name: str = "my_QCCS",
    debug=False,
) -> DeviceSetup:
    """A function to generate a DeviceSetup given a list of devices, based on standardised wiring assumptions.

    With this function, you can auto-generate a device setup based on a list of input parameters,
    making some standard assumptions about wiring configuration.

    Args:
        number_qubits: The number of qubits that the device setup shall be configured for.
        pqsc: The device id of your PQSC as a list (e.g. `["DEV10XX0"]`).
            Note: only one PQSC is possible per set-up.
        hdawg: The device id(s) of your HDAWG instruments as a list
            (e.g. `["DEV8XXX", "DEV8YYY"]`).
            Note: only 8 channel HDAWG instruments are supported at the moment.
        uhfqa: The device id(s) of your UHFQA instruments as a list
            (e.g. `["DEV2XXX", "DEV2YYY"]`).
            Note: The UHFQA cannot be used in combination with SHF devices.
        shfsg: The device id(s) of your SHFSG instruments as a list
            (e.g. `["DEV12XXX", "DEV12YYY"]`).
            Note: only 8-channel SHFSG instruments are supported at the moment.
        shfqc: The device id(s) of your SHFQC instruments as a list
            (e.g. `["DEV12XXX", "DEV12YYY"]`).
            Note: only 6 SG-channel SHFQC instruments are supported at the moment.
        shfqa: The device id(s) of your SHFQA instruments as a list
            (e.g. `["DEV12XXX", "DEV12YYY"]`).
            Note: only 4-channel SHFQA instruments are supported at the moment.
        number_multiplex: How many qubits to multiplex on a single readout line.
            If set to None, no readout multiplexing is configured.
        multiplex_drive_lines: Whether to add logical signals that are mutliplexed t othe drive lines,
            e.g. for cross-resonance or e-f driving.
        include_flux_lines: Whether to include flux lines in the setup.
        drive_only: Whether to generates a device setup without readout or acquisition lines.
        dio: A dictionary specifying the DIO connectivity between instruments,
            for setups containing HDAWG and UHFQA instruments.
            Should be of the form `{"HDAWG_ID": "UHFQA_ID"}`
        zsync: A dictionary specifying the ZSYNC configuration of any instruments attached to the PQSC.
            Should be of the form `{"DEVICE_ID": ZSYNC_PORT_NUMBER}`
        server_host: The IP address of the LabOne dataserver used to connect to the instruments.
            Defaults to "localhost".
        server_port: The port number of the LabOne dataserver used to connect to the instruments.
            Defaults to the LabOne default setting "8004".
        setup_name: The name of your setup. Defaults to "my_QCCS"
        debug: Whether to print optional debug information to console.

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

    # check input for consistency
    if uhfqa and (shfsg or shfqa or shfqc):
        raise LabOneQException(
            "Device Setup generation failed: Mixing SHF and UHF instruments in a single setup is not supported."
        )
    gen1_setup = False
    if not pqsc:
        if uhfqa and hdawg:
            if len(uhfqa) + len(hdawg) > 2:
                raise LabOneQException(
                    "Device Setup generation failed: Without PQSC, only a single UHFQA can be combined together with a single HDAWG."
                )
            else:
                print_debug("INFO: Small setup detected - HDAWG + UHFQA.")
                gen1_setup = True
                if not dio:
                    dio = {hdawg[0]: uhfqa[0]}
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
            print("INFO: Small setup detected - SHFQA only.")
    elif len(pqsc) > 1:
        raise LabOneQException(
            "Device Setup generation failed: Only a single PQSC is supported in a QCCS setup."
        )
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
    if gen1_setup:
        if include_flux_lines > 0:
            raise LabOneQException(
                "Device Setup generation failed: Flux lines are not supported for Gen1 setups."
            )
        elif multiplex_drive_lines:
            raise LabOneQException(
                "Device Setup generation failed: Cross-resonance drive lines are not supported for Gen1 setups."
            )

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
    if number_multiplex is None:
        number_multiplex = 1

    for id, instrument in enumerate(uhfqa):
        device_setup.add_instruments(UHFQA(uid=f"uhfqa_{id}", address=instrument))
        actual_multiplex = min(10, number_multiplex)
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
        instrument_list[instrument] = f"uhfqa_{id}"

    for id, instrument in enumerate(hdawg):
        device_setup.add_instruments(
            HDAWG(
                uid=f"hdawg_{id}", address=instrument, device_options="HDAWG8/MF/ME/PC"
            )
        )
        if gen1_setup:
            drive_lines.extend(
                [
                    {
                        "device": f"hdawg_{id}",
                        "port": [f"SIGOUTS/{2*it}", f"SIGOUTS/{2*it+1}"],
                    }
                    for it in range(4)
                ]
            )
            number_drive_lines += 4
        else:
            flux_lines.extend(
                [
                    {
                        "device": f"hdawg_{id}",
                        "port": f"SIGOUTS/{it}",
                    }
                    for it in range(8)
                ]
            )
            number_flux_lines += 8
        instrument_list[instrument] = f"hdawg_{id}"

    for id, instrument in enumerate(shfsg):
        device_setup.add_instruments(
            SHFSG(uid=f"shfsg_{id}", address=instrument, device_options="SHFSG8/RTR")
        )
        drive_lines.extend(
            [
                {"device": f"shfsg_{id}", "port": f"SGCHANNELS/{it}/OUTPUT"}
                for it in range(8)
            ]
        )
        number_drive_lines += 8
        instrument_list[instrument] = f"shfsg_{id}"

    for id, instrument in enumerate(shfqa):
        device_setup.add_instruments(
            SHFQA(uid=f"shfqa_{id}", address=instrument, device_options="SHFQA4/")
        )
        actual_multiplex = min(16, number_multiplex)
        readout_acquire_lines.extend(
            [
                {
                    "device": f"shfqa_{id}",
                    "port_out": "QACHANNELS/0/OUTPUT",
                    "port_in": "QACHANNELS/0/INPUT",
                    "multiplex": actual_multiplex,
                }
            ]
        )
        number_readout_acquire_lines += actual_multiplex
        instrument_list[instrument] = f"shfqa_{id}"

    for id, instrument in enumerate(shfqc):
        device_setup.add_instruments(
            SHFQC(
                uid=f"shfqc_{id}",
                address=instrument,
                device_options="SHFQC/QC6CH",
            )
        )
        drive_lines.extend(
            [
                {"device": f"shfqc_{id}", "port": f"SGCHANNELS/{it}/OUTPUT"}
                for it in range(6)
            ]
        )
        number_drive_lines += 6
        # drive_instruments[f"shfqc_{id}"] = {}
        actual_multiplex = min(16, number_multiplex)
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
        instrument_list[instrument] = f"shfqc_{id}"

    for id, instrument in enumerate(pqsc):
        device_setup.add_instruments(PQSC(uid=f"pqsc_{id}", address=instrument))
        instrument_list[instrument] = f"pqsc_{id}"

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

    # add logical signal lines for all qubits
    qubits = [f"q{it}" for it in range(number_qubits)]
    # add readout and acquire lines
    if not drive_only:
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
                    to_signal=f"{qubit}/measure_line", ports=current_readout["port_out"]
                ),
                create_connection(
                    to_signal=f"{qubit}/acquire_line", ports=current_readout["port_in"]
                ),
            )
            current_multiplex += 1

    # add drive and cr lines
    drive_index = 0
    for qubit in qubits:
        current_drive = drive_lines[drive_index]
        # advance in list of drive lines
        device_setup.add_connections(
            current_drive["device"],
            create_connection(
                to_signal=f"{qubit}/drive_line", ports=current_drive["port"]
            ),
        )
        if multiplex_drive_lines:
            device_setup.add_connections(
                current_drive["device"],
                create_connection(
                    to_signal=f"{qubit}/drive_line_ef", ports=current_drive["port"]
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
                    to_signal=f"{qubit}/flux_line", ports=current_flux["port"]
                ),
            )
            flux_index += 1

    # add DIO and ZSync connections
    if dio:
        for dio_in, dio_out in dio.items():
            device_setup.add_connections(
                instrument_list[dio_in],
                create_connection(
                    to_instrument=instrument_list[dio_out], ports="DIOS/0"
                ),
            )
    if zsync:
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

    return device_setup


def generate_device_setup_qubits(
    number_qubits: int = 6,
    pqsc: list[str] | None = None,
    hdawg: list[str] | None = None,
    shfsg: list[str] | None = None,
    shfqc: list[str] | None = None,
    shfqa: list[str] | None = None,
    uhfqa: list[str] | None = None,
    number_multiplex: int | None = 6,
    multiplex_drive_lines: bool = False,
    include_flux_lines: bool = False,
    drive_only: bool = False,
    dio: dict[str] | None = None,
    zsync: dict[str | int] | None = None,
    server_host: str = "localhost",
    server_port: str = "8004",
    setup_name: str = "my_QCCS",
    include_qubits: bool = False,
    calibrate_setup: bool = False,
):
    """A function to generate a DeviceSetup and a list of Transmon qubits
    given a list of devices, based on standardised wiring assumptions.

    With this function, you can auto-generate a device setup and a list of Transmon qubits
    based on a list of input parameters, making some standard assumptions about wiring configuration.

    Args:
        number_qubits: The number of qubits that the device setup shall be configured for.
        pqsc: The device id of your PQSC as a list (e.g. `["DEV10XX0"]`).
            Note: only one PQSC is possible per set-up.
        hdawg: The device id(s) of your HDAWG instruments as a list
            (e.g. `["DEV8XX0", "DEV8XX1"]`).
            Note: only 8 channel HDAWG instruments are supported at the moment.
        uhfqa: The device id(s) of your UHFQA instruments as a list
            (e.g. `["DEV2XX0", "DEV2XX1"]`).
            Note: The UHFQA cannot be used in combination with SHF devices.
        shfsg: The device id(s) of your SHFSG instruments as a list
            (e.g. `["DEV12XX0"]`).
            Note: only 8-channel SHFSG instruments are supported at the moment.
        shfqc: The device id(s) of your SHFQC instruments as a list
            (e.g. `["DEV12XX3"]`).
            Note: only 6 SG-channel SHFQC instruments are supported at the moment.
        shfqa: The device id(s) of your SHFQA instruments as a list
            (e.g. `["DEV12XX8"]`).
            Note: only 4-channel SHFQA instruments are supported at the moment.
        number_multiplex: How many qubits to multiplex on a single readout line.
            If set to None, no readout multiplexing is configured.
        multiplex_drive_lines: Whether to add logical signals that are mutliplexed t othe drive lines,
            e.g. for cross-resonance or e-f driving.
        include_flux_lines: Whether to include flux lines in the setup.
        drive_only: Whether to generates a device setup without readout or acquisition lines.
        dio: A dictionary specifying the DIO connectivity between instruments,
            for setups containing HDAWG and UHFQA instruments.
            Should be of the form `{"HDAWG_ID": "UHFQA_ID"}`
        zsync: A dictionary specifying the ZSYNC configuration of any instruments attached to the PQSC.
            Should be of the form `{"DEVICE_ID": ZSYNC_PORT_NUMBER}`
        server_host: The IP address of the LabOne dataserver used to connect to the instruments.
            Defaults to "localhost".
        server_port: The port number of the LabOne dataserver used to connect to the instruments.
            Defaults to the LabOne default setting "8004".
        setup_name: The name of your setup. Defaults to "my_QCCS"

    Returns:
        A LabOne Q DeviceSetup object with the specified configuration
            as well as a list of Transmon qubits configured for the given DeviceSetup.
    """

    device_setup = generate_device_setup(
        number_qubits=number_qubits,
        pqsc=pqsc,
        hdawg=hdawg,
        shfsg=shfsg,
        shfqc=shfqc,
        shfqa=shfqa,
        uhfqa=uhfqa,
        number_multiplex=number_multiplex,
        multiplex_drive_lines=multiplex_drive_lines,
        include_flux_lines=include_flux_lines,
        drive_only=drive_only,
        dio=dio,
        zsync=zsync,
        server_host=server_host,
        server_port=server_port,
        setup_name=setup_name,
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
