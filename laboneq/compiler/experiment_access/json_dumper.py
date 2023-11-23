# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import dataclasses
import typing

from laboneq.data.compilation_job import ParameterInfo

if typing.TYPE_CHECKING:
    from laboneq.compiler.experiment_access import ExperimentDAO


def dump(experiment_dao: ExperimentDAO):
    retval = {
        "$schema": "../../schemas/qccs-schema_2_5_0.json",
        "metadata": {
            "version": "2.5.0",
            "unit": {"time": "s", "frequency": "Hz", "phase": "rad"},
            "epsilon": {"time": 1e-12},
            "line_endings": "unix",
        },
    }

    device_entries = {}
    reference_clock = None
    for device in experiment_dao.devices():
        device_info = experiment_dao.device_info(device)

        device_entry = {}
        for key in ["reference_clock_source", "is_qc"]:
            if getattr(device_info, key) is not None:
                device_entry[key] = getattr(device_info, key)
        device_entry["id"] = device_info.uid
        device_entry["driver"] = device_info.device_type.name.lower()

        oscillator_ids = experiment_dao.device_oscillators(device)

        if len(oscillator_ids) > 0:
            device_entry["oscillators_list"] = [
                {"$ref": oscillator_id} for oscillator_id in oscillator_ids
            ]
        device_entries[device_entry["id"]] = device_entry
        reference_clock = experiment_dao.device_reference_clock(device)

    retval["devices"] = list(sorted(device_entries.values(), key=lambda x: x["id"]))

    connectivity_object = {}
    if experiment_dao.global_leader_device():
        connectivity_object = {
            "leader": {"$ref": experiment_dao.global_leader_device()}
        }

    if reference_clock is not None:
        connectivity_object["reference_clock"] = reference_clock
    dios = [
        {
            "leader": {"$ref": dio_connection[0]},
            "follower": {"$ref": dio_connection[1]},
        }
        for dio_connection in experiment_dao.dio_connections()
    ]
    if len(dios) > 0:
        connectivity_object["dios"] = dios

    pqscs = []

    for pqsc in experiment_dao.pqscs():
        pqsc_entry = {"device": {"$ref": pqsc}, "ports": []}

        for port_info in experiment_dao.pqsc_ports(pqsc):
            pqsc_entry["ports"].append(
                {"device": {"$ref": port_info["device"]}, "port": port_info["port"]}
            )
        pqscs.append(pqsc_entry)
    if len(pqscs) > 0:
        connectivity_object["pqscs"] = pqscs

    if len(connectivity_object.keys()) > 0:
        retval["connectivity"] = connectivity_object
    oscillator_infos = [
        experiment_dao.oscillator_info(oscillator_id)
        for oscillator_id in experiment_dao.oscillators()
    ]
    out_oscillators = []
    for oscillator_info in oscillator_infos:
        frequency = oscillator_info.frequency
        if isinstance(oscillator_info.frequency, ParameterInfo):
            frequency = {"$ref": oscillator_info.frequency.uid}
        out_oscillator_entry = {
            "id": oscillator_info.uid,
            "frequency": frequency,
            "hardware": oscillator_info.is_hardware,
        }
        out_oscillators.append(out_oscillator_entry)
    if len(out_oscillators) > 0:
        retval["oscillators"] = out_oscillators

    retval["signals"] = []
    signal_infos = [
        experiment_dao.signal_info(signal_id) for signal_id in experiment_dao.signals()
    ]

    signal_connections = []
    for signal_info in signal_infos:
        signal_entry = {
            "id": signal_info.uid,
            "signal_type": signal_info.type.value,
        }
        if signal_info.oscillator is not None:
            signal_entry["modulation"] = True
        signal_oscillator = experiment_dao.signal_oscillator(signal_info.uid)
        if signal_oscillator is not None:
            signal_entry["oscillators_list"] = [{"$ref": signal_oscillator.uid}]
        retval["signals"].append(signal_entry)

        signal_connection = {
            "signal": {"$ref": signal_info.uid},
            "device": {"$ref": signal_info.device.uid},
            "connection": {
                "type": "in" if signal_info.type.value == "integration" else "out",
                "channels": signal_info.channels,
            },
        }

        voltage_offset = experiment_dao.voltage_offset(signal_info.uid)
        if voltage_offset is not None:
            signal_connection["voltage_offset"] = voltage_offset

        mixer_calibration = experiment_dao.mixer_calibration(signal_info.uid)
        if mixer_calibration is not None:
            signal_connection["mixer_calibration"] = dataclasses.asdict(
                mixer_calibration
            )

        precompensation = experiment_dao.precompensation(signal_info.uid)
        if precompensation is not None:
            signal_connection["precompensation"] = dataclasses.asdict(precompensation)

        lo_frequency = experiment_dao.lo_frequency(signal_info.uid)
        if lo_frequency is not None:
            signal_connection["lo_frequency"] = lo_frequency

        port_mode = experiment_dao.port_mode(signal_info.uid)
        if port_mode is not None:
            signal_connection["port_mode"] = port_mode

        signal_range = experiment_dao.signal_range(signal_info.uid)
        if signal_range is not None and signal_range.value is not None:
            signal_connection["range"] = signal_range.value
        if signal_range is not None and signal_range.unit is not None:
            signal_connection["range_unit"] = signal_range.unit

        port_delay = experiment_dao.port_delay(signal_info.uid)
        if port_delay is not None:
            signal_connection["port_delay"] = port_delay

        threshold = experiment_dao.threshold(signal_info.uid)
        if threshold is not None:
            signal_connection["threshold"] = threshold

        amplitude = experiment_dao.amplitude(signal_info.uid)
        if amplitude is not None:
            signal_connection["amplitude"] = amplitude

        delay_signal = signal_info.delay_signal
        if delay_signal is not None:
            signal_connection["delay_signal"] = delay_signal

        signal_connections.append(signal_connection)

    retval["signal_connections"] = signal_connections

    pulses_list = []

    for pulse_id in experiment_dao.pulses():
        pulse = experiment_dao.pulse(pulse_id)
        pulse_entry = {"id": pulse.uid}
        if pulse.function:
            pulse_entry["function"] = pulse.function
        if pulse.amplitude is not None:
            pulse_entry["amplitude"] = pulse.amplitude
        if pulse.length is not None:
            pulse_entry["length"] = pulse.length
        if pulse.samples is not None:
            pulse_entry["samples"] = pulse.samples

        pulses_list.append(pulse_entry)
    retval["pulses"] = pulses_list

    sections = {}
    for section_id in experiment_dao.sections():
        section_info = experiment_dao.section_info(section_id)
        if section_info.uid in sections:
            # Section is reused, has already been processed
            continue

        out_section = {"id": section_info.uid}

        direct_children = experiment_dao.direct_section_children(section_id)

        direct_children = [
            experiment_dao.section_info(child_id).uid for child_id in direct_children
        ]

        if section_info.count is not None:
            out_section["repeat"] = {
                "execution_type": section_info.execution_type.value
                if section_info.execution_type
                else None,
                "count": section_info.count,
            }

            section_parameters = experiment_dao.section_parameters(section_id)
            if len(section_parameters) > 0:
                out_section["repeat"]["parameters"] = []
                for parameter in section_parameters:
                    param_object = {"id": parameter.uid}
                    keys = ["start", "step", "values"]
                    for key in keys:
                        if getattr(parameter, key) is not None:
                            param_object[key] = getattr(parameter, key)

                    out_section["repeat"]["parameters"].append(param_object)

        if len(direct_children) > 0:
            if section_info.count is not None:
                out_section["repeat"]["sections_list"] = [
                    {"$ref": child} for child in direct_children
                ]
            else:
                out_section["sections_list"] = [
                    {"$ref": child} for child in direct_children
                ]
        keys = [
            "length",
            "repetition_time",
            "play_after",
            "handle",
            "user_register",
            "state",
            "local",
        ]
        for key in keys:
            if getattr(section_info, key, None) is not None:
                out_section[key] = getattr(section_info, key)
        if section_info.reset_oscillator_phase:
            out_section["reset_oscillator_phase"] = section_info.reset_oscillator_phase
        if section_info.alignment:
            out_section["align"] = section_info.alignment.value
        if section_info.repetition_mode:
            out_section["repetition_mode"] = section_info.repetition_mode.value
        if section_info.acquisition_type:
            out_section["acquisition_types"] = [section_info.acquisition_type.value]
        if section_info.averaging_mode:
            out_section["averaging_mode"] = section_info.averaging_mode.value
        if section_info.triggers:
            out_section["trigger_output"] = [
                {
                    "signal": {"$ref": to_item["signal_id"]},
                    "state": to_item["state"],
                }
                for to_item in section_info.triggers
            ]

        signals_list = []
        for signal_id in sorted(experiment_dao.section_signals(section_id)):
            section_signal_object = {"signal": {"$ref": signal_id}}
            section_signal_pulses = []
            for section_pulse in experiment_dao.section_pulses(section_id, signal_id):
                section_signal_pulse_object = {}
                if section_pulse.pulse is not None:
                    section_signal_pulse_object["pulse"] = {
                        "$ref": section_pulse.pulse.uid
                    }
                if section_pulse.precompensation_clear:
                    section_signal_pulse_object[
                        "precompensation_clear"
                    ] = section_pulse.precompensation_clear
                for key in [
                    "amplitude",
                    "offset",
                    "increment_oscillator_phase",
                    "phase",
                    "set_oscillator_phase",
                    "length",
                ]:
                    if (val := getattr(section_pulse, key)) is not None:
                        if isinstance(val, ParameterInfo):
                            section_signal_pulse_object[key] = {"$ref": val.uid}
                        else:
                            section_signal_pulse_object[key] = val
                if section_pulse.acquire_params is not None:
                    handle = section_pulse.acquire_params.handle
                    if handle is not None:
                        section_signal_pulse_object["readout_handle"] = handle
                if section_pulse.play_pulse_parameters:
                    section_signal_pulse_object[
                        "play_pulse_parameters"
                    ] = copy.deepcopy(section_pulse.play_pulse_parameters)
                if section_pulse.pulse_pulse_parameters:
                    section_signal_pulse_object[
                        "pulse_pulse_parameters"
                    ] = copy.deepcopy(section_pulse.pulse_pulse_parameters)
                if section_pulse.pulse_group is not None:
                    section_signal_pulse_object[
                        "pulse_group"
                    ] = section_pulse.pulse_group
                markers = section_pulse.markers
                if markers is not None:
                    markers_object = {}
                    for m in markers:
                        marker_object = {}
                        for k in ["enable", "start", "length"]:
                            value = getattr(m, k)
                            if value is not None:
                                marker_object[k] = value
                        if m.pulse_id is not None:
                            marker_object["waveform"] = {"$ref": m.pulse_id}
                        markers_object[m.marker_selector] = marker_object
                    if len(markers_object) > 0:
                        section_signal_pulse_object["markers"] = markers_object

                section_signal_pulses.append(section_signal_pulse_object)

            if len(section_signal_pulses) > 0:
                section_signal_object["pulses_list"] = section_signal_pulses
            signals_list.append(section_signal_object)
        if len(signals_list) > 0:
            out_section["signals_list"] = signals_list

        sections[section_info.uid] = out_section

    retval["sections"] = list(sorted(sections.values(), key=lambda x: x["id"]))

    retval["experiment"] = {
        "sections_list": [
            {"$ref": experiment_dao.section_info(section).uid}
            for section in experiment_dao.root_sections()
        ],
        "signals_list": [{"$ref": signal_id} for signal_id in experiment_dao.signals()],
    }
    return retval
