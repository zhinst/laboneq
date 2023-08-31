# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Function to generate and fill a datastore for use with the tutorials and how_to notebooks in the LabOne Q repository
"""

import datetime

import numpy as np

from laboneq.contrib.example_helpers.example_notebook_helper import (
    create_dummy_qubit,
    create_dummy_transmon,
    generate_dummy_transmon_parameters,
)
from laboneq.contrib.example_helpers.generate_descriptor import generate_descriptor
from laboneq.dsl.device import DeviceSetup
from laboneq.implementation.data_storage.laboneq_database import DataStore


def generate_device_setup(
    pqsc=None,
    hdawg_8=None,
    uhfqa=None,
    shfsg_8=None,
    shfqa_4=None,
    shfqc_6=None,
    number_data_qubits=2,
    number_flux_lines=0,
    multiplex=False,
    number_multiplex=0,
    include_cr_lines=False,
    drive_only=False,
    dummy_dio=None,
    server_host="localhost",
    server_port="8004",
    setup_name=None,
    store_setup=False,
    datastore=None,
):
    device_setup_descriptor = generate_descriptor(
        pqsc=pqsc,
        uhfqa=uhfqa,
        hdawg_8=hdawg_8,
        shfsg_8=shfsg_8,
        shfqa_4=shfqa_4,
        shfqc_6=shfqc_6,
        multiplex=multiplex,
        number_multiplex=number_multiplex,
        number_data_qubits=number_data_qubits,
        number_flux_lines=number_flux_lines,
        drive_only=drive_only,
        include_cr_lines=include_cr_lines,
        dummy_dio=dummy_dio,
    )
    device_setup = DeviceSetup.from_descriptor(
        yaml_text=device_setup_descriptor,
        server_host=server_host,
        server_port=server_port,
        setup_name=setup_name,
    )
    if store_setup:
        datastore.store(
            data=device_setup,
            # key=setup_name,
            metadata={
                "name": setup_name,
                "type": "device_setup",
                "creation_date": datetime.datetime.now(),
            },
        )

    return device_setup


def generate_example_datastore(
    in_memory=False, path="./laboneq_data/", filename="dummy_datastore.db"
):
    # create connection to datastore
    if in_memory:
        setup_db = DataStore(":memory:")
    else:
        setup_db = DataStore(path + filename)

    # generate dummy parameter set for all qubits
    num_qubits = 24
    dummy_qubit_parameters = generate_dummy_transmon_parameters(
        number_of_qubits=num_qubits
    )

    # generic 24 qubit device setup - including flux lines
    device_setup = generate_device_setup(
        pqsc=["dev10001"],
        hdawg_8=[f"dev800{it}" for it in range(1, int(np.ceil(num_qubits / 8)) + 1)],
        shfqc_6=[f"dev1200{it}" for it in range(1, int(np.ceil(num_qubits / 6)) + 1)],
        multiplex=True,
        number_multiplex=6,
        number_data_qubits=num_qubits,
        number_flux_lines=num_qubits,
        setup_name=f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc",
        store_setup=True,
        datastore=setup_db,
    )

    # create transmon qubits from base parameters
    my_tuneable_transmons = [
        create_dummy_transmon(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(num_qubits)
    ]
    # store transmon qubits in datastore
    for it, qubit in enumerate(my_tuneable_transmons):
        setup_db.store(
            data=qubit,
            # key=f"tuneable_transmon_{it}",
            metadata={
                "name": f"tuneable_transmon_{it}",
                "type": "tuneable_transmon_qubit",
                "creation_date": datetime.datetime.now(),
            },
        )
    # create generic qubits from base parameters
    my_tuneable_qubits = [
        create_dummy_qubit(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(num_qubits)
    ]
    # store qubits in datastore
    for it, qubit in enumerate(my_tuneable_qubits):
        setup_db.store(
            data=qubit,
            # key=f"tuneable_qubit_{it}",
            metadata={
                "name": f"tuneable_qubit_{it}",
                "type": "tuneable_generic_qubit",
                "creation_date": datetime.datetime.now(),
            },
        )
    # calibrate device setup with transmon qubit parameters
    for qubit in my_tuneable_transmons:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        # key=f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc_calibrated",
        metadata={
            "name": f"{num_qubits}_tuneable_qubit_setup_shfqc_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # generic 24 qubit device setup - without flux lines
    device_setup = generate_device_setup(
        pqsc=["dev10001"],
        shfqc_6=[f"dev1200{it}" for it in range(1, int(np.ceil(num_qubits / 6)) + 1)],
        multiplex=True,
        number_multiplex=6,
        number_data_qubits=num_qubits,
        setup_name=f"{num_qubits}_qubit_setup_shfqc_pqsc",
        store_setup=True,
        datastore=setup_db,
    )

    # create transmon qubits from base parameters
    my_fixed_transmons = [
        create_dummy_transmon(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(num_qubits)
    ]
    # store transmon qubits in datastore
    for it, qubit in enumerate(my_fixed_transmons):
        setup_db.store(
            data=qubit,
            metadata={
                "name": f"fixed_transmon_{it}",
                "type": "fixed_transmon_qubit",
                "creation_date": datetime.datetime.now(),
            },
        )
    # create generic qubits from base parameters
    my_fixed_qubits = [
        create_dummy_qubit(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(num_qubits)
    ]
    # store qubits in datastore
    for it, qubit in enumerate(my_fixed_qubits):
        setup_db.store(
            data=qubit,
            metadata={
                "name": f"fixed_qubit_{it}",
                "type": "fixed_generic_qubit",
                "creation_date": datetime.datetime.now(),
            },
        )
    # calibrate device setup with transmon qubit parameters
    for qubit in my_fixed_transmons:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_fixed_qubit_setup_shfqc_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfqc
    num_qubits = 6
    device_setup = generate_device_setup(
        shfqc_6=["dev12001"],
        multiplex=True,
        number_multiplex=6,
        number_data_qubits=num_qubits,
        setup_name=f"{num_qubits}_qubit_setup_shfqc",
        store_setup=True,
        datastore=setup_db,
    )
    # calibrate device setup with qubit parameters
    for qubit in my_fixed_transmons[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_shfqc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg_shfqa_pqsc
    num_qubits = 6
    device_setup = generate_device_setup(
        pqsc=["dev10001"],
        shfsg_8=["dev12001"],
        shfqa_4=["dev12002"],
        multiplex=True,
        number_multiplex=6,
        number_data_qubits=num_qubits,
        setup_name=f"{num_qubits}_qubit_setup_shfsg_shfqa_pqsc",
        store_setup=True,
        datastore=setup_db,
    )
    # calibrate device setup with qubit parameters
    for qubit in my_fixed_transmons[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_shfsg_shfqa_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg_shfqa_hdawg_pqsc
    num_qubits = 6
    device_setup = generate_device_setup(
        pqsc=["dev10001"],
        hdawg_8=["dev8001"],
        shfsg_8=["dev12001"],
        shfqa_4=["dev12002"],
        multiplex=True,
        number_multiplex=6,
        number_data_qubits=num_qubits,
        number_flux_lines=num_qubits,
        setup_name=f"{num_qubits}_qubit_setup_shfsg_shfqa_hdawg_pqsc",
        store_setup=True,
        datastore=setup_db,
    )
    # calibrate device setup with qubit parameters
    for qubit in my_tuneable_transmons[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_shfsg_shfqa_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg_shfqa_shfqc_hdawg_pqsc
    num_qubits = 12
    device_setup = generate_device_setup(
        pqsc=["dev10001"],
        hdawg_8=["dev8001", "dev8002"],
        shfsg_8=["dev12001"],
        shfqc_6=["dev12002"],
        shfqa_4=["dev12003"],
        multiplex=True,
        number_multiplex=6,
        number_data_qubits=num_qubits,
        number_flux_lines=num_qubits,
        setup_name=f"{num_qubits}_qubit_setup_shfsg_shfqa_shfqc_hdawg_pqsc",
        store_setup=True,
        datastore=setup_db,
    )
    # calibrate device setup with qubit parameters
    for qubit in my_tuneable_transmons[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_shfsg_shfqa_shfqc_hdawg_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # hdawg_uhfqa_pqsc
    num_qubits = 2
    device_setup = generate_device_setup(
        pqsc=["dev10001"],
        hdawg_8=["dev8001"],
        uhfqa=["dev2001"],
        multiplex=True,
        number_multiplex=2,
        number_data_qubits=num_qubits,
        number_flux_lines=num_qubits,
        dummy_dio={"dev8001": "dev2001"},
        setup_name=f"{num_qubits}_qubit_setup_hdawg_uhfqa_pqsc",
        store_setup=True,
        datastore=setup_db,
    )
    # calibrate device setup with qubit parameters
    for qubit in my_tuneable_qubits[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_hdawg_uhfqa_pqsc_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # shfsg
    num_qubits = 8
    device_setup = generate_device_setup(
        shfsg_8=["dev12001"],
        number_data_qubits=num_qubits,
        drive_only=True,
        setup_name=f"{num_qubits}_qubit_setup_shfsg",
        store_setup=True,
        datastore=setup_db,
    )
    # re-create transmon qubits from base parameters - no measurement lines in this case
    my_fixed_transmons_2 = [
        create_dummy_transmon(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(num_qubits)
    ]
    # calibrate device setup with qubit parameters
    for qubit in my_fixed_transmons_2[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_shfsg_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    # hdawg
    num_qubits = 2
    device_setup = generate_device_setup(
        hdawg_8=["dev8001"],
        number_data_qubits=num_qubits,
        number_flux_lines=num_qubits,
        drive_only=True,
        setup_name=f"{num_qubits}_qubit_setup_hdawg",
        store_setup=True,
        datastore=setup_db,
    )
    # re-create transmon qubits from base parameters - no measurement lines in this case
    my_tuneable_qubits_2 = [
        create_dummy_qubit(
            it, base_parameters=dummy_qubit_parameters, device_setup=device_setup
        )
        for it in range(num_qubits)
    ]
    # calibrate device setup with qubit parameters
    for qubit in my_tuneable_qubits_2[0:num_qubits]:
        device_setup.set_calibration(qubit.calibration())
    # store calibrated device_setup in datastore
    setup_db.store(
        data=device_setup,
        metadata={
            "name": f"{num_qubits}_qubit_setup_hdawg_calibrated",
            "type": "device_setup_calibrated",
            "creation_date": datetime.datetime.now(),
        },
    )

    return setup_db


def get_first_named_entry(db: DataStore, name: str):
    key = next(db.find(metadata={"name": name}))
    return db.get(key)
