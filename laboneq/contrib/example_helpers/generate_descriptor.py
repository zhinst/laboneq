# Copyright 2020 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

"""Function to generate a descriptor from a list of instruments, making basic assumptions about the type of signals needed
"""

from __future__ import annotations
import time
from pathlib import Path

import yaml
from zhinst.toolkit.driver.devices import PQSC
from zhinst.toolkit.session import Session


def generate_descriptor(
    pqsc: list | None = None,  # ["DEV10XX0"]
    hdawg_4: list | None = None,  # ["DEV8XX0", "DEV8XX1"],
    hdawg_8: list | None = None,
    uhfqa: list | None = None,
    shfsg_4: list | None = None,  # ["DEV12XX0", "DEV12XX1"],
    shfsg_8: list | None = None,
    shfqc_2: list | None = None,
    shfqc_4: list | None = None,
    shfqc_6: list | None = None,  # ["DEV12XX2"],
    shfqa_2: list | None = None,
    shfqa_4: list | None = None,  # ["DEV12XX3"],
    number_data_qubits=2,
    number_flux_lines=0,
    multiplex=False,
    number_multiplex=0,
    include_cr_lines=False,
    drive_only=False,
    readout_only=False,
    internal_clock=False,
    save=False,
    filename="yaml_descriptor",
    get_zsync=False,
    get_dio=False,
    dummy_dio: dict | None = None,  # {"DEV8XX0":"DEV2XX0"}
    ip_address: str = "localhost",
):
    """A function to generate a descriptor given a list of devices based on wiring assumptions.

    With this function, you can generate a descriptor file quickly. This descriptor will produce
    a string or (optionally) a YAML file using the instruments provided, logical signal groups
    derived from the number of specified qubits, and logical signal lines based off of the options
    chosen by the user, e.g., whether to multiplex the readout, how many qubits to multiplex per
    readout line, etc. The generated descriptor therefore specifies to the user how their qubits
    should be wired to the Zurich Instruments devices. If the user prefers to specify the wiring
    of each qubit individually, this should be done by editing the YAML string manually.

    Args:
        pqsc: The device id of your PQSC as a list (e.g. `["DEV10XX0"]`).
            Note: only one PQSC is possible per set-up.
        hdawg_4: The device id(s) of your 4-channel HDAWG instruments as a list
            (e.g. `["DEV8XX0", "DEV8XX1"]`).
        hdawg_8: The device id(s) of your 8-channel HDAWG instruments as a list
            (e.g. `["DEV8XX2", "DEV8XX3", "DEV8XX4"]`).
        uhfqa: The device id(s) of your UHFQA instruments as a list
            (e.g. `["DEV2XX0", "DEV2XX1"]`).
            Note: The UHFQA cannot be used combined with SHF devices.
        shfsg_4: The device id(s) of your 4-channel SHFSG instruments as a list
            (e.g. `["DEV12XX0"]`).
        shfsg_8: The device id(s) of your 8-channel SHFSG instruments as a list
            (e.g. `["DEV12XX1", "DEV12XX2"]`).
        shfqc_2: The device id(s) of your 2 SG-channel SHFQC instruments as a list
            (e.g. `["DEV12XX3"]`).
        shfqc_4: The device id(s) of your 4 SG-channel SHFQC instruments as a list
            (e.g. `["DEV12XX4"]`).
        shfqc_6: The device id(s) of your 6 SG-channel SHFQC instruments as a list
            (e.g. `["DEV12XX5", "DEV12XX6", "DEV12XX7"]`).
        shfqa_2: The device id(s) of your 2-channel SHFQA instruments as a list
            (e.g. `["DEV12XX8"]`).
        shfqa_4: The device id(s) of your 4-channel SHFQA instruments as a list
            (e.g. `["DEV12XX9"]`).
        number_data_qubits: The number of qubits to create logical signal groups for.
        number_flux_lines: The total number of flux lines, using the assumption that there
            is no more than one flux line per qubit.
        multiplex: If True, qubits will be multiplexed according to number_multiplex.
        number_multiplex: The number of qubits to multiplex per physical quantum analyzer channel.
        drive_only: If True, generates a descriptor without measure or acquisition lines.
        readout_only: If True, generates a descriptor without drive or flux lines.
        internal_clock: If True, uses the internal PQSC clock.
            Note: the PQSC internal clock can not be used combined with the UHFQA.
        save: If True, creates a Descriptor file in the active directory and
            saves a YAML file with the name specified in filename.
        filename: The file name to give to the YAML descriptor (e.g. `"yaml_descriptor"`).
        get_zsync: If True, starts a Session to communicate with the PQSC and
            listed devices to determine the connections of the ZSync cables.
        get_dio: If True, starts a Session to determine the connections of HDAWG
            to UHFQA instruments via DIO cables.
        dummy_dio: Allows the user to specify a dictionary with a DIO connection
            without querying the instruments with the HDAWG as the key and UHFQA as
            the value
            (e.g. `{"DEV8XX0": "DEV2XX0"}`).
        ip_address: The IP address needed to connect to the instruments if using
            get_zsync or get_dio.

    Returns:
        A string in YAML format and, optionally, a YAML file.
    """

    # Make combined lists and dicts
    pqsc_list = []
    hd_list = []
    uhf_list = []
    sg_list = []
    qc_list = []
    qa_list = []
    all_list = []
    devid_uid = {}

    if pqsc is not None:
        pqsc_list.extend(pqsc)
    if hdawg_4 is not None:
        for i in hdawg_4:
            hd_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"HDAWG_{i}"
    if hdawg_8 is not None:
        for i in hdawg_8:
            hd_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"HDAWG_{i}"
    if uhfqa is not None:
        for i in uhfqa:
            uhf_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"UHFQA_{i}"
    if shfsg_4 is not None:
        for i in shfsg_4:
            sg_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFSG_{i}"
    if shfsg_8 is not None:
        for i in shfsg_8:
            sg_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFSG_{i}"
    if shfqc_2 is not None:
        for i in shfqc_2:
            qc_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFQC_{i}"
    if shfqc_4 is not None:
        for i in shfqc_4:
            qc_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFQC_{i}"
    if shfqc_6 is not None:
        for i in shfqc_6:
            qc_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFQC_{i}"
    if shfqa_2 is not None:
        for i in shfqa_2:
            qa_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFQA_{i}"
    if shfqa_4 is not None:
        for i in shfqa_4:
            qa_list.append(i)
            all_list.append(i)
            devid_uid[i] = f"SHFQA_{i}"

    # Get numbers of instruments from user lists
    number_hdawg_4 = len(hdawg_4) if hdawg_4 is not None else 0
    number_hdawg_8 = len(hdawg_8) if hdawg_8 is not None else 0

    number_uhfqa = len(uhfqa) if uhfqa is not None else 0

    number_shfsg_4 = len(shfsg_4) if shfsg_4 is not None else 0
    number_shfsg_8 = len(shfsg_8) if shfsg_8 is not None else 0

    number_shfqc_2 = len(shfqc_2) if shfqc_2 is not None else 0
    number_shfqc_4 = len(shfqc_4) if shfqc_4 is not None else 0
    number_shfqc_6 = len(shfqc_6) if shfqc_6 is not None else 0

    number_shfqa_2 = len(shfqa_2) if shfqa_2 is not None else 0
    number_shfqa_4 = len(shfqa_4) if shfqa_4 is not None else 0

    # Specify instrument channels
    n_qa_uhfqa = 1
    n_qa_shfqa_2 = 2
    n_qa_shfqa_4 = 4
    n_qa_shfqc_6 = 1
    n_qa_shfqc_4 = 1
    n_qa_shfqc_2 = 1

    n_ch_hdawg_4 = 4
    n_ch_hdawg_8 = 8

    n_iq_shfsg_4 = 4
    n_iq_shfsg_8 = 8
    n_iq_shfqc_2 = 2
    n_iq_shfqc_4 = 4
    n_iq_shfqc_6 = 6

    if uhfqa is None:
        number_qa_lines = (
            n_qa_shfqa_2 * number_shfqa_2
            + n_qa_shfqa_4 * number_shfqa_4
            + n_qa_shfqc_6 * number_shfqc_6
            + n_qa_shfqc_4 * number_shfqc_4
            + n_qa_shfqc_2 * number_shfqc_2
        )
    elif uhfqa is not None:
        number_qa_lines = n_qa_uhfqa * number_uhfqa

    # Check if enough output lines are present
    number_shf_output_lines = (
        n_iq_shfqc_2 * number_shfqc_2
        + n_iq_shfqc_4 * number_shfqc_4
        + n_iq_shfqc_6 * number_shfqc_6
        + n_iq_shfsg_4 * number_shfsg_4
        + n_iq_shfsg_8 * number_shfsg_8
    )

    number_hd_lines = n_ch_hdawg_4 * number_hdawg_4 + n_ch_hdawg_8 * number_hdawg_8

    # Check remainong control lines
    after_shf_iq = number_shf_output_lines - number_data_qubits
    # print(f"After SHF IQ: {after_shf_iq}")

    # Check remaining flux lines
    after_hd_flux = number_hd_lines - number_flux_lines
    # print(f"After HD flux: {after_hd_flux}")

    if after_shf_iq < 0:
        leftover_output = int(after_hd_flux / 2) + after_shf_iq
    elif after_shf_iq > 0 and after_hd_flux < 0:
        leftover_output = after_shf_iq + after_hd_flux
    elif after_hd_flux == 0:
        leftover_output = after_shf_iq
    elif after_shf_iq == 0:
        leftover_output = after_hd_flux
    elif after_shf_iq > 0 and after_hd_flux > 0:
        leftover_output = after_shf_iq + after_hd_flux

    # Check if enough analyzer channels are present
    if multiplex is False:
        tot_qa_lines = number_qa_lines
    elif multiplex is True:
        tot_qa_lines = number_multiplex * number_qa_lines

    # Compatibility check
    if len(all_list) > 1 and not pqsc:
        print(
            """\
A PQSC is required to synchronize multiple instruments.
If you are using only a single HDAWG and UHFQA, please see
https://docs.zhinst.com/labone_q_user_manual/concepts/set_up_equipment.html
for how to set them up without a PQSC.
        """
        )
        return
    if len(all_list) > 18:
        print(
            "The PQSC only supports up to 18 instruments. Please reduce number of instruments."
        )
        return
    elif pqsc is not None and len(pqsc) > 1:
        print("Cannot have more than one PQSC in a descriptor!")
        return
    elif len(all_list) != len(set(all_list)):
        print("Duplicate Device IDs! Please check your device lists.")
        return
    elif leftover_output < 0 and readout_only is False:
        print("Not enough output lines for number of qubits specified!")
        return
    elif uhfqa is not None and number_multiplex >= 11:
        print("Can't multiplex more than 10 qubits on a UHFQA!")
        return
    elif uhfqa is not None and internal_clock is True:
        print("PQSC internal clock not supported with a UHFQA!")
        return
    elif number_multiplex >= 17:
        print("Can't multiplex more than 16 qubits on a SHFQA or QC analyzer channel!")
        return
    elif tot_qa_lines < number_data_qubits and drive_only is False:
        print("Not enough available readout lines!")
        return
    elif (
        uhfqa is not None
        and (shfqa_2 or shfqa_4 or shfqc_2 or shfqc_4 or shfqc_6 or shfsg_4 or shfsg_8)
        is not None
    ):
        print("UHFQA not supported in combination with SHF Instruments.")
        return
    elif (
        get_dio is True
        and (shfqa_2 or shfqa_4 or shfqc_2 or shfqc_4 or shfqc_6 or shfsg_4 or shfsg_8)
        is not None
    ):
        print("Get DIO not supported with SHF Instruments.")
        return
    elif get_dio and dummy_dio:
        print("Can't use get_dio and dummy_dio together!")
        return

    # Create instrument dictionary
    def generate_instrument_list(instrument, instrument_name):
        instrument_list = [
            {"address": entry, "uid": f"{instrument_name}_{entry}"}
            for entry in instrument
        ]
        return instrument_list

    instrument_dict = {
        "PQSC": generate_instrument_list(pqsc_list, "PQSC") if pqsc else None,
        "HDAWG": generate_instrument_list(hd_list, "HDAWG") if hd_list else None,
        "UHFQA": generate_instrument_list(uhf_list, "UHFQA") if uhf_list else None,
        "SHFSG": generate_instrument_list(sg_list, "SHFSG") if sg_list else None,
        "SHFQC": generate_instrument_list(qc_list, "SHFQC") if qc_list else None,
        "SHFQA": generate_instrument_list(qa_list, "SHFQA") if qa_list else None,
    }

    clean_instruments_dict = {
        "instruments": {k: v for k, v in instrument_dict.items() if v is not None}
    }

    # Assign logical signals and ports
    signal_and_port_dict = {}
    # IQ Line Outputs
    current_qubit = 0
    if readout_only is False:
        if shfqc_6 is not None:
            i_shfqc_6, i_qc_ch_6 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_6[i_shfqc_6]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": f"SGCHANNELS/{i_qc_ch_6}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line_ef",
                        "ports": f"SGCHANNELS/{i_qc_ch_6}/OUTPUT",
                    }
                )
                if include_cr_lines:
                    sig_dict.append(
                        {
                            "iq_signal": f"q{i}/drive_line_cr",
                            "ports": f"SGCHANNELS/{i_qc_ch_6}/OUTPUT",
                        }
                    )
                i_qc_ch_6 += 1
                current_qubit += 1
                if i_qc_ch_6 >= n_iq_shfqc_6:
                    i_qc_ch_6 = 0
                    i_shfqc_6 += 1
                if i_shfqc_6 == len(shfqc_6):
                    break
        if shfqc_4 is not None:
            i_shfqc_4, i_qc_ch_4 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_4[i_shfqc_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": f"SGCHANNELS/{i_qc_ch_4}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line_ef",
                        "ports": f"SGCHANNELS/{i_qc_ch_4}/OUTPUT",
                    }
                )
                if include_cr_lines:
                    sig_dict.append(
                        {
                            "iq_signal": f"q{i}/drive_line_cr",
                            "ports": f"SGCHANNELS/{i_qc_ch_4}/OUTPUT",
                        }
                    )
                i_qc_ch_4 += 1
                current_qubit += 1
                if i_qc_ch_4 >= n_iq_shfqc_4:
                    i_qc_ch_4 = 0
                    i_shfqc_4 += 1
                if i_shfqc_4 == len(shfqc_4):
                    break
        if shfqc_2 is not None:
            i_shfqc_2, i_qc_ch_2 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_2[i_shfqc_2]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": f"SGCHANNELS/{i_qc_ch_2}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line_ef",
                        "ports": f"SGCHANNELS/{i_qc_ch_2}/OUTPUT",
                    }
                )
                if include_cr_lines:
                    sig_dict.append(
                        {
                            "iq_signal": f"q{i}/drive_line_cr",
                            "ports": f"SGCHANNELS/{i_qc_ch_2}/OUTPUT",
                        }
                    )
                i_qc_ch_2 += 1
                current_qubit += 1
                if i_qc_ch_2 >= n_iq_shfqc_2:
                    i_qc_ch_2 = 0
                    i_shfqc_2 += 1
                if i_shfqc_2 == len(shfqc_2):
                    break
        if shfsg_8 is not None:
            i_shfsg_8, i_sg_ch_8 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFSG_{shfsg_8[i_shfsg_8]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": f"SGCHANNELS/{i_sg_ch_8}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line_ef",
                        "ports": f"SGCHANNELS/{i_sg_ch_8}/OUTPUT",
                    }
                )
                if include_cr_lines:
                    sig_dict.append(
                        {
                            "iq_signal": f"q{i}/drive_line_cr",
                            "ports": f"SGCHANNELS/{i_sg_ch_8}/OUTPUT",
                        }
                    )
                i_sg_ch_8 += 1
                current_qubit += 1
                if i_sg_ch_8 >= n_iq_shfsg_8:
                    i_sg_ch_8 = 0
                    i_shfsg_8 += 1
                if i_shfsg_8 == len(shfsg_8):
                    break
        if shfsg_4 is not None:
            i_shfsg_4, i_sg_ch_4 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFSG_{shfsg_4[i_shfsg_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": f"SGCHANNELS/{i_sg_ch_4}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line_ef",
                        "ports": f"SGCHANNELS/{i_sg_ch_4}/OUTPUT",
                    }
                )
                if include_cr_lines:
                    sig_dict.append(
                        {
                            "iq_signal": f"q{i}/drive_line_cr",
                            "ports": f"SGCHANNELS/{i_sg_ch_4}/OUTPUT",
                        }
                    )
                i_sg_ch_4 += 1
                current_qubit += 1
                if i_sg_ch_4 >= n_iq_shfsg_4:
                    i_sg_ch_4 = 0
                    i_shfsg_4 += 1
                if i_shfsg_4 == len(shfsg_4):
                    break
        if hdawg_8 is not None:
            i_hdawg_8, i_hd_ch_8 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"HDAWG_{hdawg_8[i_hdawg_8]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": [f"SIGOUTS/{i_hd_ch_8}", f"SIGOUTS/{i_hd_ch_8+1}"],
                    }
                )
                i_hd_ch_8 += 2
                current_qubit += 1
                if i_hd_ch_8 >= n_ch_hdawg_8:
                    i_hd_ch_8 = 0
                    i_hdawg_8 += 1
                if i_hdawg_8 == len(hdawg_8):
                    break
        if hdawg_4 is not None:
            i_hdawg_4, i_hd_ch_4 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"HDAWG_{hdawg_4[i_hdawg_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/drive_line",
                        "ports": [f"SIGOUTS/{i_hd_ch_4}", f"SIGOUTS/{i_hd_ch_4+1}"],
                    }
                )
                i_hd_ch_4 += 2
                current_qubit += 1
                if i_hd_ch_4 >= n_ch_hdawg_4:
                    i_hd_ch_4 = 0
                    i_hdawg_4 += 1
                if i_hdawg_4 == len(hdawg_4):
                    break

    # Flux Lines
    current_qubit = 0
    if readout_only is False:
        if hdawg_8 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"HDAWG_{hdawg_8[i_hdawg_8]}", []
                )
                sig_dict.append(
                    {
                        "rf_signal": f"q{i}/flux_line",
                        "ports": f"SIGOUTS/{i_hd_ch_8}",
                    }
                )
                i_hd_ch_8 += 1
                current_qubit += 1
                if i_hd_ch_8 >= n_ch_hdawg_8:
                    i_hd_ch_8 = 0
                    i_hdawg_8 += 1
                if i_hdawg_8 == len(hdawg_8):
                    break
        if hdawg_4 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"HDAWG_{hdawg_4[i_hdawg_4]}", []
                )
                sig_dict.append(
                    {
                        "rf_signal": f"q{i}/flux_line",
                        "ports": f"SIGOUTS/{i_hd_ch_4}",
                    }
                )
                i_hd_ch_4 += 1
                current_qubit += 1
                if i_hd_ch_4 >= n_ch_hdawg_4:
                    i_hd_ch_4 = 0
                    i_hdawg_4 += 1
                if i_hdawg_4 == len(hdawg_4):
                    break
        if shfsg_8 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFSG_{shfsg_8[i_shfsg_8]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/flux_line",
                        "ports": f"SGCHANNELS/{i_sg_ch_8}/OUTPUT",
                    }
                )
                i_sg_ch_8 += 1
                current_qubit += 1
                if i_sg_ch_8 >= n_iq_shfsg_8:
                    i_sg_ch_8 = 0
                    i_shfsg_8 += 1
                if i_shfsg_8 == len(shfsg_8):
                    break
        if shfsg_4 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFSG_{shfsg_4[i_shfsg_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/flux_line",
                        "ports": f"SGCHANNELS/{i_sg_ch_4}/OUTPUT",
                    }
                )
                i_sg_ch_4 += 1
                current_qubit += 1
                if i_sg_ch_4 >= n_iq_shfsg_4:
                    i_sg_ch_4 = 0
                    i_shfsg_4 += 1
                if i_shfsg_4 == len(shfsg_4):
                    break
        if shfqc_6 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_6[i_shfqc_6]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/flux_line",
                        "ports": f"SGCHANNELS/{i_qc_ch_6}/OUTPUT",
                    }
                )
                i_qc_ch_6 += 1
                current_qubit += 1
                if i_qc_ch_6 >= n_iq_shfqc_6:
                    i_qc_ch_6 = 0
                    i_shfqc_6 += 1
                if i_shfqc_6 == len(shfqc_6):
                    break
        if shfqc_4 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_4[i_shfqc_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/flux_line",
                        "ports": f"SGCHANNELS/{i_qc_ch_4}/OUTPUT",
                    }
                )
                i_qc_ch_4 += 1
                current_qubit += 1
                if i_qc_ch_4 >= n_iq_shfqc_4:
                    i_qc_ch_4 = 0
                    i_shfqc_4 += 1
                if i_shfqc_4 == len(shfqc_4):
                    break
        if shfqc_2 is not None:
            for i in range(current_qubit, number_flux_lines):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_2[i_shfqc_2]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/flux_line",
                        "ports": f"SGCHANNELS/{i_qc_ch_2}/OUTPUT",
                    }
                )
                i_qc_ch_2 += 1
                current_qubit += 1
                if i_qc_ch_2 >= n_iq_shfqc_2:
                    i_qc_ch_2 = 0
                    i_shfqc_2 += 1
                if i_shfqc_2 == len(shfqc_2):
                    break

    # QA Lines
    current_qubit = 0
    # Without multiplexed readout
    if not multiplex and not drive_only:
        if shfqc_6 is not None:
            i_shfqc_6, i_qc_qa_6 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_6[i_shfqc_6]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qc_qa_6}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qc_qa_6}/INPUT",
                    }
                )
                i_qc_qa_6 += 1
                current_qubit += 1
                if i_qc_qa_6 >= n_qa_shfqc_6:
                    i_qc_qa_6 = 0
                    i_shfqc_6 += 1
                if i_shfqc_6 == len(shfqc_6):
                    break
        if shfqc_4 is not None:
            i_shfqc_4, i_qc_qa_4 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_4[i_shfqc_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qc_qa_4}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qc_qa_4}/INPUT",
                    }
                )
                i_qc_qa_4 += 1
                current_qubit += 1
                if i_qc_qa_4 >= n_qa_shfqc_4:
                    i_qc_qa_4 = 0
                    i_shfqc_4 += 1
                if i_shfqc_4 == len(shfqc_4):
                    break
        if shfqc_2 is not None:
            i_shfqc_2, i_qc_qa_2 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_2[i_shfqc_2]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qc_qa_2}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qc_qa_2}/INPUT",
                    }
                )
                i_qc_qa_2 += 1
                current_qubit += 1
                if i_qc_qa_2 >= n_qa_shfqc_2:
                    i_qc_qa_2 = 0
                    i_shfqc_2 += 1
                if i_shfqc_2 == len(shfqc_2):
                    break
        if shfqa_4 is not None:
            i_shfqa_4, i_qa_ch_4 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQA_{shfqa_4[i_shfqa_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qa_ch_4}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qa_ch_4}/INPUT",
                    }
                )
                i_qa_ch_4 += 1
                current_qubit += 1
                if i_qa_ch_4 >= n_qa_shfqa_4:
                    i_qa_ch_4 = 0
                    i_shfqa_4 += 1
                if i_shfqa_4 == len(shfqa_4):
                    break
        if shfqa_2 is not None:
            i_shfqa_2, i_qa_ch_2 = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQA_{shfqa_2[i_shfqa_2]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qa_ch_2}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qa_ch_2}/INPUT",
                    }
                )
                i_qa_ch_2 += 1
                current_qubit += 1
                if i_qa_ch_2 >= n_qa_shfqa_2:
                    i_qa_ch_2 = 0
                    i_shfqa_2 += 1
                if i_shfqa_2 == len(shfqa_2):
                    break
        if uhfqa is not None:
            i_uhfqa, i_uhfqa_ch = 0, 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"UHFQA_{uhfqa[i_uhfqa]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": [f"SIGOUTS/{i_uhfqa_ch}", f"SIGOUTS/{i_uhfqa_ch+1}"],
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                    }
                )
                i_uhfqa_ch += 2
                current_qubit += 1
                if i_uhfqa_ch >= n_qa_uhfqa:
                    i_uhfqa_ch = 0
                    i_uhfqa += 1
                if i_uhfqa == len(uhfqa):
                    break
    # With multiplexed readout
    if multiplex and not drive_only:
        if shfqc_6 is not None:
            i_shfqc_6, i_qc_qa_6 = 0, 0
            multiplex_number = 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_6[i_shfqc_6]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qc_qa_6}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qc_qa_6}/INPUT",
                    }
                )
                current_qubit += 1
                multiplex_number += 1
                if multiplex_number >= number_multiplex:
                    multiplex_number = 0
                    i_qc_qa_6 += 1
                    if i_qc_qa_6 >= n_qa_shfqc_6:
                        i_qc_qa_6 = 0
                        i_shfqc_6 += 1
                    if i_shfqc_6 == len(shfqc_6):
                        break
        if shfqc_4 is not None:
            i_shfqc_4, i_qc_qa_4 = 0, 0
            multiplex_number = 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_4[i_shfqc_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qc_qa_4}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qc_qa_4}/INPUT",
                    }
                )
                current_qubit += 1
                multiplex_number += 1
                if multiplex_number >= number_multiplex:
                    multiplex_number = 0
                    i_qc_qa_4 += 1
                    if i_qc_qa_4 >= n_qa_shfqc_4:
                        i_qc_qa_4 = 0
                        i_shfqc_4 += 1
                    if i_shfqc_4 == len(shfqc_4):
                        break
        if shfqc_2 is not None:
            i_shfqc_2, i_qc_qa_2 = 0, 0
            multiplex_number = 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQC_{shfqc_2[i_shfqc_2]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qc_qa_2}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qc_qa_2}/INPUT",
                    }
                )
                current_qubit += 1
                multiplex_number += 1
                if multiplex_number >= number_multiplex:
                    multiplex_number = 0
                    i_qc_qa_2 += 1
                    if i_qc_qa_2 >= n_qa_shfqc_2:
                        i_qc_qa_2 = 0
                        i_shfqc_2 += 1
                    if i_shfqc_2 == len(shfqc_2):
                        break
        if shfqa_4 is not None:
            i_shfqa_4, i_qa_ch_4 = 0, 0
            multiplex_number = 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQA_{shfqa_4[i_shfqa_4]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qa_ch_4}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qa_ch_4}/INPUT",
                    }
                )
                current_qubit += 1
                multiplex_number += 1
                if multiplex_number >= number_multiplex:
                    multiplex_number = 0
                    i_qa_ch_4 += 1
                    if i_qa_ch_4 >= n_qa_shfqa_4:
                        i_qa_ch_4 = 0
                        i_shfqa_4 += 1
                    if i_shfqa_4 == len(shfqa_4):
                        break
        if shfqa_2 is not None:
            i_shfqa_2, i_qa_ch_2 = 0, 0
            multiplex_number = 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"SHFQA_{shfqa_2[i_shfqa_2]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": f"QACHANNELS/{i_qa_ch_2}/OUTPUT",
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                        "ports": f"QACHANNELS/{i_qa_ch_2}/INPUT",
                    }
                )
                current_qubit += 1
                multiplex_number += 1
                if multiplex_number >= number_multiplex:
                    multiplex_number = 0
                    i_qa_ch_2 += 1
                    if i_qa_ch_2 >= n_qa_shfqa_2:
                        i_qa_ch_2 = 0
                        i_shfqa_2 += 1
                    if i_shfqa_2 == len(shfqa_2):
                        break
        if uhfqa is not None:
            i_uhfqa, i_uhfqa_ch = 0, 0
            multiplex_number = 0
            for i in range(current_qubit, number_data_qubits):
                sig_dict = signal_and_port_dict.setdefault(
                    f"UHFQA_{uhfqa[i_uhfqa]}", []
                )
                sig_dict.append(
                    {
                        "iq_signal": f"q{i}/measure_line",
                        "ports": [f"SIGOUTS/{i_uhfqa_ch}", f"SIGOUTS/{i_uhfqa_ch+1}"],
                    }
                )
                sig_dict.append(
                    {
                        "acquire_signal": f"q{i}/acquire_line",
                    }
                )
                current_qubit += 1
                multiplex_number += 1
                if multiplex_number >= number_multiplex:
                    multiplex_number = 0
                    i_uhfqa_ch += 2
                    if i_uhfqa_ch >= n_qa_uhfqa:
                        i_uhfqa_ch = 0
                        i_uhfqa += 1
                    if i_uhfqa == len(uhfqa):
                        break

    # PQSC connections
    if pqsc is not None and not get_zsync:
        for i, k in enumerate(devid_uid):
            sig_dict = signal_and_port_dict.setdefault(f"PQSC_{pqsc[0]}", [])
            if devid_uid[k].split("_")[0] == "UHFQA":
                continue
            sig_dict.append(
                {
                    "to": f"{devid_uid[k]}",
                    "port": f"ZSYNCS/{i}",
                }
            )
        if internal_clock is True:
            sig_dict.append("internal_clock_signal")

    if get_zsync or get_dio:
        session = Session(ip_address)
        if pqsc is not None and get_zsync:
            device_pqsc = session.connect_device(pqsc[0])
            print("Checking PQSC Connections...")
            for k in devid_uid:
                session_device = session.connect_device(devid_uid[k].split("_")[1])
                if "SHF" in session_device.device_type:
                    print(devid_uid[k].split("_")[1])
                    session_device.system.clocks.referenceclock.in_.source(2)
                if "HDAWG" in session_device.device_type:
                    print(devid_uid[k].split("_")[1])
                    session_device.system.clocks.referenceclock.source(2)
                if "UHFQA" in session_device.device_type:
                    continue
                time.sleep(2)
                sig_dict = signal_and_port_dict.setdefault(f"PQSC_{pqsc[0]}", [])
                sig_dict.append(
                    {
                        "to": f"{devid_uid[k]}",
                        "port": f"ZSYNCS/{PQSC.find_zsync_worker_port(self=device_pqsc, device=session_device)}",
                    }
                )
            if internal_clock is True:
                sig_dict.append("internal_clock_signal")

        # HD and UHFQA DIOS
        if get_dio:
            time.sleep(2)
            if hdawg_8 is not None and uhfqa is not None:
                for hd in hdawg_8:
                    sig_dict = signal_and_port_dict.setdefault(f"HDAWG_{hd}", [])
                    device_hd = session.connect_device(hd)
                    device_hd.dios[0].output(int(hd.split("V")[1]))
                    device_hd.dios[0].drive(15)
                    for uhf in uhfqa:
                        device_uhfqa = session.connect_device(uhf)
                        time.sleep(4)
                        codeword = device_uhfqa.dios[0].input()["dio"][0]
                        if codeword == int(hd.split("V")[1]):
                            print(f"{hd} connected to {uhf} via DIO")
                            sig_dict.append(
                                {
                                    "to": f"UHFQA_{uhf}",
                                    "port": "DIOS/0",
                                }
                            )
                    device_hd.dios[0].drive(0)
            if hdawg_4 is not None and uhfqa is not None:
                for hd in hdawg_4:
                    sig_dict = signal_and_port_dict.setdefault(f"HDAWG_{hd}", [])
                    device_hd = session.connect_device(hd)
                    device_hd.dios[0].output(int(hd.split("V")[1]))
                    device_hd.dios[0].drive(15)
                    for uhf in uhfqa:
                        device_uhfqa = session.connect_device(uhf)
                        time.sleep(4)
                        codeword = device_uhfqa.dios[0].input()["dio"][0]
                        if codeword == int(hd.split("V")[1]):
                            print(f"{hd} connected to {uhf} via DIO")
                            sig_dict.append(
                                {
                                    "to": f"UHFQA_{uhf}",
                                    "port": "DIOS/0",
                                }
                            )
                    device_hd.dios[0].drive(0)
        with session.set_transaction():
            session.disconnect_device(pqsc[0])
            for device in all_list:
                session.disconnect_device(device)

    if hdawg_8 is not None and uhfqa is not None and dummy_dio:
        for hd in hdawg_8:
            sig_dict = signal_and_port_dict.setdefault(f"HDAWG_{hd}", [])
            if hd in str(dummy_dio):
                sig_dict.append(
                    {
                        "to": f"UHFQA_{dummy_dio[hd]}",
                        "port": "DIOS/0",
                    }
                )

    if hdawg_4 is not None and uhfqa is not None and dummy_dio:
        for hd in hdawg_4:
            sig_dict = signal_and_port_dict.setdefault(f"HDAWG_{hd}", [])
            if hd in str(dummy_dio):
                sig_dict.append(
                    {
                        "to": f"UHFQA_{dummy_dio[hd]}",
                        "port": "DIOS/0",
                    }
                )

    clean_connections_dict = {
        "connections": {k: v for k, v in signal_and_port_dict.items() if v is not None}
    }

    # Generate final dictionary and YAML
    yaml_dict = {}

    yaml_dict.update(clean_connections_dict)
    yaml_dict.update(clean_instruments_dict)

    yaml_final = yaml.safe_dump(yaml_dict)

    if save is True:
        Path("Descriptors").mkdir(parents=True, exist_ok=True)
        with open(f"./Descriptors/{filename}.yaml", "w") as file:
            yaml.safe_dump(yaml_dict, file)

    return yaml_final
