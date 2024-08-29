# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Function to generate and fill a datastore for use with the tutorials and how_to notebooks in the LabOne Q repository"""

import datetime

from laboneq.implementation.data_storage.laboneq_database import DataStore

from .generate_device_setup import generate_device_setup_qubits


def generate_example_datastore(
    in_memory=False,
    path="./laboneq_data/",
    filename="dummy_datastore.db",
    server_host="localhost",
    include_qubits=True,
) -> DataStore:
    """A function to generate a DataStore for use with the public LabOne Q examples.

    The generated Datastore is loaded with a selection of DeviceSetups and Qubits preconfigured to be used
    with the public example notebooks.

    Args:
        in_memory: Whether to generate the Datastore in-memory instead of on the filesystem.
        path: Path to the location of the Datastore on the filesystem. Only applies if `in_memory=False`.
        filename: Filename of the generated Datastore on the filesystem. Only applies if `in_memory=False`.
        server_host: IP address of the LabOne dataserver used to communicate to the instruments of the QCCS.
            Defaults to "localhost".
        include_qubits: Whether to include qubits directly in the DeviceSetup, through the `DeviceSetup.qubits` property.
            If set to "False", no qubits will be generated.

    Returns:
        A LabOne Q Datastore loaded with a selection of device setups for use with the public LabOne Q examples.
    """

    # create (connection to) datastore
    if in_memory:
        setup_db = DataStore(":memory:")
    else:
        setup_db = DataStore(path + filename)

    # specify your device ids and parameters here
    pqsc = [{"serial": "dev10001"}]
    hdawg = [
        {"serial": "dev8001", "zsync": 1, "dio": "dev2001"},
        {"serial": "dev8002", "zsync": 2},
        {"serial": "dev8003", "zsync": 3},
    ]
    shfqc = [
        {"serial": "dev12003", "zsync": 4, "readout_multiplex": 6},
        {"serial": "dev12004", "zsync": 5, "readout_multiplex": 6},
        {"serial": "dev12005", "zsync": 6, "readout_multiplex": 6},
        {"serial": "dev12006", "zsync": 7, "readout_multiplex": 6},
    ]
    shfsg = [{"serial": "dev12001", "zsync": 8}]
    shfqa = [{"serial": "dev12002", "zsync": 9, "readout_multiplex": 6}]
    uhfqa = [{"serial": "dev2001"}]

    # generic 6 qubit device setup - including flux lines
    num_qubits = 6
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        hdawg=hdawg[:1],
        shfqc=shfqc[:1],
        include_flux_lines=True,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # large 24 qubit device setup - including flux lines
    num_qubits = 24
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        hdawg=hdawg[:3],
        shfqc=shfqc,
        include_flux_lines=True,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # generic 6 qubit device setup - without flux lines
    num_qubits = 6
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        hdawg=hdawg[:1],
        shfqc=shfqc[:1],
        include_flux_lines=False,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_fixed_qubit_setup_shfqc_hdawg_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_shfqc_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfqc standalone
    num_qubits = 6
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        shfqc=shfqc[:1],
        include_flux_lines=False,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_fixed_qubit_setup_shfqc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_shfqc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg_shfqa_pqsc
    num_qubits = 6
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        shfsg=shfsg,
        shfqa=shfqa,
        include_flux_lines=False,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_fixed_qubit_setup_shfsg_shfqa_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_shfsg_shfqa_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg_shfqa_hdawg_pqsc
    num_qubits = 6
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        hdawg=hdawg[:1],
        shfsg=shfsg,
        shfqa=shfqa,
        include_flux_lines=True,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_tuneable_qubit_setup_shfsg_shfqa_hdawg_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_tuneable_qubit_setup_shfsg_shfqa_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg_shfqa_shfqc_hdawg_pqsc
    num_qubits = 12
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        hdawg=hdawg[:2],
        shfsg=shfsg,
        shfqa=shfqa,
        shfqc=shfqc[:1],
        include_flux_lines=True,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_tuneable_qubit_setup_shfsg_shfqa_shfqc_hdawg_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_tuneable_qubit_setup_shfsg_shfqa_shfqc_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # hdawg_uhfqa_pqsc
    num_qubits = 2
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        pqsc=pqsc,
        hdawg=hdawg[:1],
        uhfqa=uhfqa,
        include_flux_lines=False,
        multiplex_drive_lines=False,
        setup_name=f"{num_qubits}_fixed_qubit_setup_hdawg_uhfqa_pqsc",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_hdawg_uhfqa_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg standalone
    num_qubits = 8
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        shfsg=shfsg,
        drive_only=True,
        include_flux_lines=False,
        multiplex_drive_lines=True,
        setup_name=f"{num_qubits}_fixed_qubit_setup_shfsg",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_shfsg_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # hdawg standalone
    num_qubits = 4
    device_setup, _ = generate_device_setup_qubits(
        number_qubits=num_qubits,
        hdawg=hdawg[:1],
        drive_only=True,
        include_flux_lines=False,
        multiplex_drive_lines=False,
        setup_name=f"{num_qubits}_fixed_qubit_setup_hdawg",
        server_host=server_host,
        include_qubits=include_qubits,
        calibrate_setup=True,
    )
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_hdawg_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    return setup_db


def get_first_named_entry(db: DataStore, name: str):
    """Return the first entry in a LabOne Q Datastore whose metadata contains the matching name"""
    key = next(db.find(metadata={"name": name}))
    return db.get(key)
