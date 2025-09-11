# Copyright 2025 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Functions to generate a device setup and a list of qubits from parameters, to use with the tutorials and how_to notebooks in the LabOne Q repository"""

from __future__ import annotations

import math
import random

from laboneq import __version__ as laboneq_version
from laboneq.contrib.example_helpers.example_notebook_helper import (
    create_dummy_transmon,
    generate_dummy_transmon_parameters,
)
from laboneq.contrib.example_helpers.generate_device_setup import (
    generate_device_setup,
)
from laboneq.dsl.quantum import QPU
from laboneq.simple import DeviceSetup, dsl
from laboneq_applications.qpu_types.tunable_transmon import (
    TunableTransmonOperations,
    TunableTransmonQubit,
    TunableTransmonQubitParameters,
)


# to compare version from string versions
def versiontuple(v):
    return tuple(map(int, (v.split("."))))


# function to create a transmon qubit object from entries in a parameter dictionary
def create_dummy_tunable_transmons(
    number_of_qubits,
    device_setup,
    drive_centre_frequency=4e9,
    readout_centre_frequency=6e9,
):
    qubit_uids = [f"q{it}" for it in range(number_of_qubits)]
    # parameters
    parameter_dict = {}
    for qubit_id in qubit_uids:
        resonance_frequency_ge = (
            drive_centre_frequency + random.uniform(-2.5, 2.5) * 100e6
        )
        parameter_dict[qubit_id] = TunableTransmonQubitParameters(
            resonance_frequency_ge=resonance_frequency_ge,
            resonance_frequency_ef=resonance_frequency_ge - 250e6,
            readout_resonator_frequency=readout_centre_frequency
            + random.uniform(-2.5, 2.5) * 100e6,
            drive_lo_frequency=drive_centre_frequency,
            readout_lo_frequency=readout_centre_frequency,
            readout_range_out=5,
            readout_range_in=10,
            readout_integration_delay=40e-9,
            user_defined={
                "amplitude_pi": 0.66,
                "pulse_length": 100e-9,
                "readout_length": 2e-6,
                "readout_amplitude": 0.6,
                "reset_delay_length": 100e-9,
            },
        )
    qubits = TunableTransmonQubit.from_device_setup(
        device_setup=device_setup,
        qubit_uids=qubit_uids,
        parameters=parameter_dict,
    )
    return qubits


# generate a device setup from a list of Instrument objects
def generate_device_setup_benchmarks(
    gen2: bool = True,
    number_qubits: int = 6,
    qhub: list | None = None,
    pqsc: list | None = None,
    hdawg: list | None = None,
    shfsg: list | None = None,
    shfqc: list | None = None,
    shfqa: list | None = None,
    uhfqa: list | None = None,
    multiplex_drive_lines: bool = False,
    include_flux_lines: bool = False,
    drive_only: bool = False,
    readout_only: bool = False,
    server_host: str = "localhost",
    server_port: str = "8004",
    setup_name: str = "my_QCCS",
    debug: bool = False,
) -> DeviceSetup:
    """A function to generate a DeviceSetup given a list of devices, based on standardised wiring assumptions.

    With this function, you can auto-generate a device setup based on a list of input parameters,
    making some standard assumptions about wiring configuration.

    Args:
        gen2: Whether to generate a device setup with generation 2 instruments.
            Defaults to True.
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
        multiplex_drive_lines: Whether to add logical signals that are multiplexed to the drive lines,
            e.g. for cross-resonance or e-f driving.
            Defaults to False.
        include_flux_lines: Whether to include flux lines in the setup.
            Defaults to False.
        drive_only: Whether to generate a device setup without readout or acquisition lines.
            Defaults to False.
        readout_only: Whether to generate a device setup without any drive or flux lines.
            Defaults to False.
        server_host: The IP address of the LabOne dataserver used to connect to the instruments.
            Defaults to "localhost".
        server_port: The port number of the LabOne dataserver used to connect to the instruments.
            Defaults to the LabOne default setting "8004".
        setup_name: The name of your setup.
            Defaults to "my_QCCS"
        debug: Whether to print optional debug information to console.
            Defaults to False.

    Returns:
        A LabOne Q DeviceSetup object with the specified configuration.
    """
    return generate_device_setup(
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
        debug=debug,
    )


def generate_device_setup_qubits_benchmarks(
    gen2: bool = True,
    number_qubits: int = 6,
    qhub: list | None = None,
    pqsc: list | None = None,
    hdawg: list | None = None,
    shfsg: list | None = None,
    shfqc: list | None = None,
    shfqa: list | None = None,
    uhfqa: list | None = None,
    multiplex_drive_lines: bool = False,
    include_flux_lines: bool = False,
    drive_only: bool = False,
    readout_only: bool = False,
    server_host: str = "localhost",
    server_port: str = "8004",
    setup_name: str = "my_QCCS",
    include_qubits: bool = True,
    calibrate_setup: bool = True,
    debug: bool = False,
):
    """A function to generate a DeviceSetup and a list of Transmon qubits
    given a list of devices, based on standardised wiring assumptions.

    With this function, you can auto-generate a device setup and a list of Transmon qubits
    based on a list of input parameters, making some standard assumptions about wiring configuration.

    Args:
        gen2: Whether to generate a device setup with generation 2 instruments.
            Defaults to True.
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
        multiplex_drive_lines: Whether to add logical signals that are multiplexed to the drive lines,
            e.g. for cross-resonance or e-f driving.
            Defaults to False.
        include_flux_lines: Whether to include flux lines in the setup.
            Defaults to False.
        drive_only: Whether to generate a device setup without readout or acquisition lines.
            Defaults to False.
        readout_only: Whether to generate a device setup without any drive or flux lines.
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
        debug: Whether to print optional debug information to console.
            Defaults to False.

    Returns:
        A LabOne Q DeviceSetup object with the specified configuration
            as well as a list of Transmon qubits configured for the given DeviceSetup.
    """

    device_setup = generate_device_setup_benchmarks(
        gen2=gen2,
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
        debug=debug,
    )
    # newer qubit types are only available in newer LabOne Q
    if versiontuple(laboneq_version) > (2, 51, 0):
        qubits = create_dummy_tunable_transmons(
            number_of_qubits=number_qubits, device_setup=device_setup
        )
    else:
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
        device_setup.qubits = {q.uid: q for q in qubits}
    if calibrate_setup:
        for qubit in qubits:
            device_setup.set_calibration(qubit.calibration())
    return device_setup, qubits


def create_benchmark_device_setup(
    gen2: bool = True,
    number_of_qubits: int = 108,
    include_flux_lines: bool = True,
    calibrate_setup: bool = False,
):
    """Function to create a device setup with a specified number of qubits."""
    if include_flux_lines:
        num_hdawg = math.ceil(number_of_qubits / 8)
    else:
        num_hdawg = 0
    num_shfqc = math.ceil(number_of_qubits / 6)
    shfqc = [
        {
            "serial": f"DEV1200{it + 1}",
            "number_of_channels": 6,
            "readout_multiplex": 6,
            "options": "SHFQC/QC6CH",
        }
        for it in range(num_shfqc)
    ]
    hdawg = [
        {
            "serial": f"DEV800{it + 1}]",
            "number_of_channels": 8,
            "options": "HDAWG8/MF/ME/SKW/PC",
        }
        for it in range(num_hdawg)
    ]

    num_zsync = num_hdawg + num_shfqc
    if num_zsync > 56 and gen2:
        print("Number of qubits specified cannot be supported on a Gen2 setup")
        print(f"required number of zSync ports: {num_zsync}")
        return

    device_setup, _ = generate_device_setup_qubits_benchmarks(
        gen2=gen2,
        number_qubits=number_of_qubits,
        qhub=[{"serial": "DEV10001"}],
        hdawg=hdawg,
        shfqc=shfqc,
        include_flux_lines=include_flux_lines,
        include_qubits=True,
        calibrate_setup=calibrate_setup,
        server_host="localhost",
        setup_name=f"{number_of_qubits}_qubit_setup_gen2",
    )

    return device_setup


def create_two_qubit_Operations():
    qOps = TunableTransmonOperations()
    qOps["sx"] = qOps.x90
    qOps["x"] = qOps.x180
    qOps["reset"] = qOps.active_reset

    @qOps.register
    def cx(
        self, q_control: TunableTransmonQubit, q_target: TunableTransmonQubit
    ) -> None:
        """An operation implementing a cx gate on two qubits.

        The controlled X gate is implemented using a cross-resonance gate.
        """
        cx_id = f"cx_{q_control.uid}_{q_target.uid}"

        # define cancellation pulses for target and control
        cancellation_control_n = dsl.create_pulse(
            {"function": "gaussian_square"}, name="CR-"
        )
        cancellation_control_p = dsl.create_pulse(
            {"function": "gaussian_square"}, name="CR+"
        )
        cancellation_target_p = dsl.create_pulse(
            {"function": "gaussian_square"}, name="q1+"
        )
        cancellation_target_n = dsl.create_pulse(
            {"function": "gaussian_square"}, name="q1-"
        )

        # play X pulses on both target and control
        with dsl.section(name=f"{cx_id}_x_both") as x180_both:
            self.x180(q_control)
            self.x180(q_target)

        # First cross-resonance component
        with dsl.section(
            name=f"{cx_id}_canc_p", play_after=x180_both.uid
        ) as cancellation_p:
            dsl.play(signal=q_target.signals["drive"], pulse=cancellation_target_p)
            dsl.play(signal=q_control.signals["flux"], pulse=cancellation_control_n)

        # play X pulse on control
        x180_control = self.x180(q_control)
        x180_control.play_after = cancellation_p.uid

        # Second cross-resonance component
        with dsl.section(name=f"cx_{cx_id}_canc_n", play_after=x180_control.uid):
            dsl.play(signal=q_target.signals["drive"], pulse=cancellation_target_n)
            dsl.play(signal=q_control.signals["flux"], pulse=cancellation_control_p)

    return qOps


def create_benchmark_device_setup_qpu(
    gen2: bool = True,
    number_of_qubits: int = 108,
    include_flux_lines: bool = True,
    calibrate_setup: bool = False,
):
    device_setup = create_benchmark_device_setup(
        gen2=gen2,
        number_of_qubits=number_of_qubits,
        include_flux_lines=include_flux_lines,
        calibrate_setup=calibrate_setup,
    )

    qOps = create_two_qubit_Operations()

    qpu = QPU(device_setup.qubits, quantum_operations=qOps)

    return device_setup, qpu
