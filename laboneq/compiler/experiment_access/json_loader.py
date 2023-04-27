# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

import copy
import json
import logging
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Tuple

from jsonschema.validators import validator_for

from laboneq.compiler.experiment_access.acquire_info import AcquireInfo
from laboneq.compiler.experiment_access.loader_base import LoaderBase
from laboneq.compiler.experiment_access.marker import Marker
from laboneq.compiler.experiment_access.pulse_def import PulseDef
from laboneq.compiler.experiment_access.section_info import SectionInfo
from laboneq.compiler.experiment_access.section_signal_pulse import SectionSignalPulse
from laboneq.core.exceptions import LabOneQException
from laboneq.core.types.enums import AcquisitionType

_logger = logging.getLogger(__name__)


def find_value_or_parameter_dict(
    pulse_ref: Dict[str, Any], name: str, types: Tuple[type, ...]
):
    param = None
    value = pulse_ref.get(name)
    if value is not None and not isinstance(value, types):
        param = value.get("$ref")
        value = None
    return value, param


class JsonLoader(LoaderBase):

    _validator = None

    def load(self, experiment: Dict):
        self._load_servers(experiment)
        self._load_devices(experiment)
        self._load_oscillator(experiment)
        self._load_connectivity(experiment)
        self._load_signals(experiment)
        self._load_signal_connections(experiment)
        self._load_pulses(experiment)
        self._load_sections(experiment)

    def _load_servers(self, experiment):
        for server in experiment["servers"]:
            self.add_server(
                server["id"],
                server.get("host"),
                server.get("port"),
                server.get("api_level"),
            )

    def _load_devices(self, experiment):
        for device in sorted(experiment["devices"], key=lambda x: x["id"]):
            if "server" in device:
                server = device["server"]["$ref"]
            else:
                server = None
            if "driver" in device:
                driver = device["driver"]
            else:
                driver = device["device_type"]

            if "serial" in device:
                serial = device["serial"]
            else:
                serial = None

            if "interface" in device:
                interface = device["interface"]
            else:
                interface = None

            if (
                "reference_clock_source" in device
                and device["reference_clock_source"] is not None
            ):
                reference_clock_source = device["reference_clock_source"]
            else:
                reference_clock_source = None

            is_qc = device.get("is_qc")

            self.add_device(
                device["id"],
                driver,
                serial,
                server,
                interface,
                reference_clock_source=reference_clock_source,
                is_qc=is_qc,
            )

            if "oscillators_list" in device:
                for oscillator_ref in device["oscillators_list"]:
                    self.add_device_oscillator(device["id"], oscillator_ref["$ref"])

    def _load_oscillator(self, experiment):
        if "oscillators" in experiment:
            for oscillator in experiment["oscillators"]:
                frequency = None
                frequency_param = None
                if "frequency" in oscillator:
                    frequency = oscillator["frequency"]
                    if not isinstance(frequency, float) and not isinstance(
                        frequency, int
                    ):
                        if frequency is not None and "$ref" in frequency:
                            frequency_param = frequency["$ref"]
                        frequency = None

                self.add_oscillator(
                    oscillator["id"],
                    frequency,
                    frequency_param,
                    bool(oscillator["hardware"]),
                )

    def _load_connectivity(self, experiment):
        if "connectivity" in experiment:
            if "dios" in experiment["connectivity"]:
                for dio in experiment["connectivity"]["dios"]:
                    self._dios.append((dio["leader"]["$ref"], dio["follower"]["$ref"]))
            if "leader" in experiment["connectivity"]:

                leader_device_id = experiment["connectivity"]["leader"]["$ref"]
                self._devices[leader_device_id]["is_global_leader"] = True

            if "reference_clock" in experiment["connectivity"]:
                reference_clock = experiment["connectivity"]["reference_clock"]
                for device in self._devices.values():
                    if device["device_type"] in {"hdawg", "uhfqa", "pqsc"}:
                        device["reference_clock"] = reference_clock

            if "pqscs" in experiment["connectivity"]:
                pqscs = experiment["connectivity"]["pqscs"]
                for pqsc in pqscs:
                    pqsc_device_id = pqsc["device"]["$ref"]
                    if "ports" in pqsc:
                        for port in pqsc["ports"]:
                            self._pqsc_ports.append(
                                (pqsc_device_id, port["device"]["$ref"], port["port"])
                            )

    def _load_signals(self, experiment):
        for signal in sorted(experiment["signals"], key=lambda s: s["id"]):
            self.add_signal(
                signal["id"],
                signal["signal_type"],
                modulation=bool(signal.get("modulation")),
            )
            if "oscillators_list" in signal:
                for oscillator_ref in signal["oscillators_list"]:
                    oscillator_id = oscillator_ref["$ref"]
                    self.add_signal_oscillator(signal["id"], oscillator_id)

    def _load_signal_connections(self, experiment):
        for connection in experiment["signal_connections"]:
            try:
                voltage_offset = copy.deepcopy(connection["voltage_offset"])
            except KeyError:
                voltage_offset = None
            try:
                mixer_calibration = copy.deepcopy(connection["mixer_calibration"])
            except KeyError:
                mixer_calibration = None
            try:
                precompensation = copy.deepcopy(connection["precompensation"])
            except KeyError:
                precompensation = None
            range = connection.get("range")
            range_unit = connection.get("range_unit")
            lo_frequency = connection.get("lo_frequency")
            port_delay = connection.get("port_delay")
            delay_signal = connection.get("delay_signal")

            self.add_signal_connection(
                connection["signal"]["$ref"],
                {
                    "signal_id": connection["signal"]["$ref"],
                    "device_id": connection["device"]["$ref"],
                    "connection_type": connection["connection"]["type"],
                    "channels": connection["connection"]["channels"],
                    "voltage_offset": voltage_offset,
                    "mixer_calibration": mixer_calibration,
                    "precompensation": precompensation,
                    "lo_frequency": lo_frequency,
                    "range": range,
                    "range_unit": range_unit,
                    "port_delay": port_delay,
                    "delay_signal": delay_signal,
                    "port_mode": None,
                    "threshold": None,
                    "amplifier_pump": None,
                },
            )

    def _load_pulses(self, experiment):
        for pulse in experiment["pulses"]:
            samples = pulse.get("samples", None)

            amplitude, amplitude_param = find_value_or_parameter_dict(
                pulse, "amplitude", (int, float, complex)
            )

            self.add_pulse(
                pulse["id"],
                PulseDef(
                    id=pulse["id"],
                    function=pulse.get("function"),
                    length=pulse.get("length"),
                    amplitude=amplitude,
                    amplitude_param=amplitude_param,
                    play_mode=pulse.get("play_mode"),
                    samples=samples,
                ),
            )

    def _load_sections(self, experiment):
        self._root_sections = sorted(
            [s["$ref"] for s in experiment["experiment"]["sections_list"]]
        )

        duplicate_sections = {
            s: count
            for (s, count) in Counter(s["id"] for s in experiment["sections"]).items()
            if count > 1
        }
        if len(duplicate_sections) == 1:
            section = next(iter(duplicate_sections.keys()))
            raise LabOneQException(f"Duplicate section id '{section}' in experiment")
        elif len(duplicate_sections) > 0:
            sections = ", ".join(f"'{s}'" for s in duplicate_sections.keys())
            raise LabOneQException(f"Duplicate section ids {sections} in experiment")

        sections_proto = {section["id"]: section for section in experiment["sections"]}
        section_reuse_counter = {}

        sections_to_process = deque((s, None) for s in self._root_sections)

        while len(sections_to_process) > 0:
            section_name, parent_instance = sections_to_process.pop()
            if section_name in section_reuse_counter:
                section_reuse_counter[section_name] += 1
                instance_id = f"{section_name}_{section_reuse_counter[section_name]}"
            else:
                section_reuse_counter[section_name] = 0
                instance_id = section_name

            section = sections_proto[section_name]

            if parent_instance is not None:
                self.add_section_child(parent_instance, instance_id)

            sections_list = None
            if "repeat" in section and "sections_list" in section["repeat"]:
                sections_list = section["repeat"]["sections_list"]
            if "sections_list" in section:
                sections_list = section["sections_list"]
            if sections_list is not None:
                for child_section_ref in sections_list:
                    sections_to_process.appendleft(
                        (child_section_ref["$ref"], instance_id)
                    )
            has_repeat = False
            execution_type = None
            length = None
            count: int = 1
            averaging_type = None

            if "repeat" in section:
                has_repeat = True
                execution_type = section["repeat"]["execution_type"]
                if "averaging_type" in section["repeat"]:
                    averaging_type = section["repeat"]["averaging_type"]

                count = int(section["repeat"]["count"])
                if "parameters" in section["repeat"]:
                    for parameter in section["repeat"]["parameters"]:
                        values = None
                        if parameter.get("values") is not None:
                            values = copy.deepcopy(list(parameter["values"]))

                        self.add_section_parameter(
                            instance_id,
                            parameter["id"],
                            parameter.get("start"),
                            parameter.get("step"),
                            values,
                        )

            acquisition_types = section.get("acquisition_types")
            # backwards-compatibility: "acquisition_types" field was previously named "trigger"
            acquisition_types = acquisition_types or section.get("trigger")
            if self.acquisition_type is None and acquisition_types is not None:
                self.acquisition_type = AcquisitionType(acquisition_types[0])

            align = None
            if "align" in section:
                align = section["align"]

            on_system_grid = False
            if "on_system_grid" in section:
                on_system_grid = section["on_system_grid"]

            reset_oscillator_phase = False
            if "reset_oscillator_phase" in section:
                reset_oscillator_phase = section["reset_oscillator_phase"]

            handle = None
            if "handle" in section:
                handle = section["handle"]

            state = None
            if "state" in section:
                state = section["state"]

            local = None
            if "local" in section:
                local = section["local"]

            if "length" in section:
                length = section["length"]

            averaging_mode = None
            if "averaging_mode" in section:
                averaging_mode = section["averaging_mode"]

            repetition_time = None
            if "repetition_time" in section:
                repetition_time = section["repetition_time"]

            repetition_mode = None
            if "repetition_mode" in section:
                repetition_mode = section["repetition_mode"]

            trigger_output = []
            for to_item in section.get("trigger_output", ()):
                trigger_signal = to_item["signal"]["$ref"]
                trigger_state = to_item["state"]
                trigger_output.append(
                    {"signal_id": trigger_signal, "state": trigger_state}
                )
                v = self._signal_trigger.get(trigger_signal, 0)
                self._signal_trigger[trigger_signal] = v | trigger_state

            self.add_section(
                instance_id,
                SectionInfo(
                    section_id=instance_id,
                    section_display_name=section["id"],
                    has_repeat=has_repeat,
                    execution_type=execution_type,
                    count=count,
                    acquisition_types=acquisition_types,
                    averaging_type=averaging_type,
                    align=align,
                    on_system_grid=on_system_grid,
                    length=length,
                    averaging_mode=averaging_mode,
                    repetition_mode=repetition_mode,
                    repetition_time=repetition_time,
                    play_after=section.get("play_after"),
                    reset_oscillator_phase=reset_oscillator_phase,
                    trigger_output=trigger_output,
                    handle=handle,
                    state=state,
                    local=local,
                ),
            )

            if "signals_list" in section:
                for signals_list_entry in section["signals_list"]:
                    signal_id = signals_list_entry["signal"]["$ref"]
                    self.add_section_signal(instance_id, signal_id)

                    if "pulses_list" in signals_list_entry:
                        for pulse_ref in signals_list_entry["pulses_list"]:
                            (
                                pulse_offset,
                                pulse_offset_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "offset", (int, float)
                            )
                            (
                                pulse_amplitude,
                                pulse_amplitude_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "amplitude", (int, float, complex)
                            )
                            (
                                pulse_increment,
                                pulse_increment_oscillator_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "increment_oscillator_phase", (int, float)
                            )
                            (
                                pulse_set_oscillator_phase,
                                pulse_set_oscillator_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "set_oscillator_phase", (int, float)
                            )
                            (
                                pulse_phase,
                                pulse_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "phase", (int, float)
                            )
                            (
                                resulting_pulse_instance_length,
                                resulting_pulse_instance_length_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "length", (int, float)
                            )

                            precompensation_clear = pulse_ref.get(
                                "precompensation_clear", False
                            )

                            pulse_id = None
                            if pulse_ref.get("pulse") is not None:
                                pulse_id = pulse_ref["pulse"]["$ref"]

                            acquire_params = None
                            signal_type = self._signals[signal_id]["signal_type"]
                            if signal_type == "integration":
                                acquire_params = AcquireInfo(
                                    handle=pulse_ref.get("readout_handle"),
                                    acquisition_type=getattr(
                                        self.acquisition_type, "value", None
                                    ),
                                )
                            markers = []
                            if "markers" in pulse_ref:
                                for k, v in pulse_ref["markers"].items():
                                    marker_pulse_id = None
                                    pulse_ref = v.get("waveform")
                                    if pulse_ref is not None:
                                        marker_pulse_id = pulse_ref["$ref"]

                                    markers.append(
                                        Marker(
                                            k,
                                            enable=v.get("enable"),
                                            start=v.get("start"),
                                            length=v.get("length"),
                                            pulse_id=marker_pulse_id,
                                        )
                                    )
                                    self.add_signal_marker(signal_id, k)

                            new_ssp = SectionSignalPulse(
                                pulse_id=pulse_id,
                                signal_id=signal_id,
                                offset=pulse_offset,
                                offset_param=pulse_offset_param,
                                amplitude=pulse_amplitude,
                                amplitude_param=pulse_amplitude_param,
                                length=resulting_pulse_instance_length,
                                length_param=resulting_pulse_instance_length_param,
                                acquire_params=acquire_params,
                                phase=pulse_phase,
                                phase_param=pulse_phase_param,
                                increment_oscillator_phase=pulse_increment,
                                increment_oscillator_phase_param=pulse_increment_oscillator_phase_param,
                                set_oscillator_phase=pulse_set_oscillator_phase,
                                set_oscillator_phase_param=pulse_set_oscillator_phase_param,
                                play_pulse_parameters=None,
                                pulse_pulse_parameters=None,
                                precompensation_clear=precompensation_clear,
                                markers=markers,
                            )
                            self.add_section_signal_pulse(
                                instance_id, signal_id, new_ssp
                            )

    @staticmethod
    def experiment_json_schema():
        with open(
            os.path.join(
                Path(__file__).parent.parent.absolute(), "qccs-schema_2_5_0.json"
            )
        ) as schema_file:
            return json.load(schema_file)

    @classmethod
    def schema_validator(cls):
        if cls._validator is None:
            schema = cls.experiment_json_schema()
            validator_cls = validator_for(schema)
            validator_cls.check_schema(schema)
            validator = validator_cls(schema)
            cls._validator = validator
        return cls._validator
