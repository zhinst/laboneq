# Copyright 2022 Zurich Instruments AG
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import json
import logging
import os
from collections import Counter, deque
from pathlib import Path
from typing import Any, Dict, Tuple

from jsonschema.validators import validator_for

from laboneq.compiler.experiment_access.loader_base import LoaderBase
from laboneq.core.exceptions import LabOneQException
from laboneq.data.calibration import ExponentialCompensation, HighPassCompensation
from laboneq.data.compilation_job import (
    AcquireInfo,
    FollowerInfo,
    Marker,
    MixerCalibrationInfo,
    PrecompensationInfo,
    PulseDef,
    SectionInfo,
    SectionSignalPulse,
    SignalRange,
)
from laboneq.data.experiment_description import (
    AcquisitionType,
    AveragingMode,
    ExecutionType,
    RepetitionMode,
    SectionAlignment,
)

_logger = logging.getLogger(__name__)


def find_value_or_parameter_dict(
    pulse_ref: Dict[str, Any], name: str, types: Tuple[type, ...]
) -> tuple[float | None, str | None]:
    param = None
    value = pulse_ref.get(name)
    if value is not None and not isinstance(value, types):
        param = value.get("$ref")
        value = None
    return value, param


class JsonLoader(LoaderBase):
    _validator = None

    def load(self, experiment: Dict):
        self._load_devices(experiment)
        self._load_oscillator(experiment)
        self._load_connectivity(experiment)
        self._load_signals(experiment)
        self._load_signal_connections(experiment)
        self._load_pulses(experiment)
        self._load_sections(experiment)

    def _load_devices(self, experiment):
        for device in sorted(experiment["devices"], key=lambda x: x["id"]):
            if "driver" in device:
                driver = device["driver"]
            else:
                driver = device["device_type"]

            reference_clock_source = device.get("reference_clock_source")

            is_qc = device.get("is_qc")

            self.add_device(
                device_id=device["id"],
                device_type=driver,
                reference_clock_source=reference_clock_source,
                is_qc=is_qc,
            )

            if "oscillators_list" in device:
                for oscillator_ref in device["oscillators_list"]:
                    self.add_device_oscillator(device["id"], oscillator_ref["$ref"])

    def _load_oscillator(self, experiment):
        for oscillator in experiment.get("oscillators", []):
            if (frequency := oscillator.get("frequency")) is None:
                continue
            if not isinstance(frequency, (int, float)):
                if "$ref" in frequency:
                    frequency = self._get_or_create_parameter(frequency["$ref"])
                else:
                    frequency = None

            self.add_oscillator(
                oscillator["id"],
                frequency,
                bool(oscillator["hardware"]),
            )

    def _load_connectivity(self, experiment):
        if "connectivity" in experiment:
            for dio in experiment["connectivity"].get("dios", {}):
                leader = self._devices[dio["leader"]["$ref"]]
                follower = self._devices[dio["follower"]["$ref"]]
                leader.followers.append(FollowerInfo(follower, 0))
            if "leader" in experiment["connectivity"]:
                leader_device_id = experiment["connectivity"]["leader"]["$ref"]
                self.global_leader_device_id = leader_device_id

            if "reference_clock" in experiment["connectivity"]:
                reference_clock = experiment["connectivity"]["reference_clock"]
                for device in self._devices.values():
                    if device.device_type.value in {"hdawg", "uhfqa", "pqsc"}:
                        device.reference_clock = reference_clock

            for pqsc in experiment["connectivity"].get("pqscs", {}):
                pqsc_device_id = pqsc["device"]["$ref"]
                for port in pqsc.get("ports", ()):
                    self._devices[pqsc_device_id].followers.append(
                        FollowerInfo(
                            self._devices[port["device"]["$ref"]], port["port"]
                        )
                    )

    def _load_signals(self, experiment):
        for signal in sorted(experiment["signals"], key=lambda s: s["id"]):
            self.add_signal(signal["id"], signal["signal_type"])
            if "oscillators_list" in signal:
                for oscillator_ref in signal["oscillators_list"]:
                    oscillator_id = oscillator_ref["$ref"]
                    self.add_signal_oscillator(signal["id"], oscillator_id)

    def _load_signal_connections(self, experiment):
        for connection in experiment["signal_connections"]:
            voltage_offset = copy.deepcopy(connection.get("voltage_offset"))
            mixer_calibration_dict = connection.get("mixer_calibration")
            if mixer_calibration_dict is not None:
                mixer_calibration = MixerCalibrationInfo(
                    voltage_offsets=mixer_calibration_dict.get("voltage_offsets"),
                    correction_matrix=mixer_calibration_dict.get("correction_matrix"),
                )
            else:
                mixer_calibration = None
            precompensation_dict = connection.get("precompensation")
            if precompensation_dict is not None:
                exponential = precompensation_dict.get("exponential")
                if exponential:
                    exponential = [ExponentialCompensation(**e) for e in exponential]
                high_pass = precompensation_dict.get("high_pass")
                if high_pass:
                    high_pass = HighPassCompensation(**high_pass)
                bounce = precompensation_dict.get("bounce")
                if bounce:
                    bounce = HighPassCompensation(**bounce)
                FIR = precompensation_dict.get("FIR")
                if FIR:
                    FIR = HighPassCompensation(**FIR)

                precompensation = PrecompensationInfo(
                    exponential, high_pass, bounce, FIR
                )
            else:
                precompensation = None
            range = connection.get("range")
            range_unit = connection.get("range_unit")
            if range is not None or range_unit is not None:
                range = SignalRange(range, range_unit)
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
                    "amplitude": None,
                    "amplifier_pump": None,
                },
            )

    def _load_pulses(self, experiment):
        for pulse in experiment["pulses"]:
            samples = pulse.get("samples", None)

            amplitude, amplitude_param = find_value_or_parameter_dict(
                pulse, "amplitude", (int, float, complex)
            )
            if amplitude_param is not None:
                raise LabOneQException(
                    f"Amplitude of pulse '{pulse.uid}' cannot be a parameter."
                    f" To sweep the amplitude, pass the parameter in the"
                    f" corresponding `play()` command."
                )

            self.add_pulse(
                pulse["id"],
                PulseDef(
                    uid=pulse["id"],
                    function=pulse.get("function"),
                    length=pulse.get("length"),
                    amplitude=amplitude,
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
            execution_type = None
            length = None
            count: int | None = None

            if "repeat" in section:
                execution_type = ExecutionType(section["repeat"]["execution_type"])

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

            acquisition_type = None
            for field in ["acquisition_types", "trigger"]:
                # backwards-compatibility: "acquisition_types" field was previously named "trigger"
                acquisition_type = section.get(field, [acquisition_type])[0]
                if acquisition_type is not None:
                    acquisition_type = AcquisitionType(acquisition_type)
            if self.acquisition_type is None and acquisition_type is not None:
                self.acquisition_type = acquisition_type

            align = None
            if "align" in section:
                align = SectionAlignment(section["align"])

            on_system_grid = False
            if "on_system_grid" in section:
                on_system_grid = section["on_system_grid"]

            reset_oscillator_phase = False
            if "reset_oscillator_phase" in section:
                reset_oscillator_phase = section["reset_oscillator_phase"]

            handle = None
            if "handle" in section:
                handle = section["handle"]

            user_register = None
            if "user_register" in section:
                user_register = section["user_register"]

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
                averaging_mode = AveragingMode(section["averaging_mode"])

            repetition_time = None
            if "repetition_time" in section:
                repetition_time = section["repetition_time"]

            repetition_mode = None
            if "repetition_mode" in section:
                repetition_mode = RepetitionMode(section["repetition_mode"])

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
                    uid=instance_id,
                    execution_type=execution_type,
                    count=count,
                    chunk_count=1,
                    acquisition_type=acquisition_type,
                    alignment=align,
                    on_system_grid=on_system_grid,
                    length=length,
                    averaging_mode=averaging_mode,
                    repetition_mode=repetition_mode,
                    repetition_time=repetition_time,
                    play_after=section.get("play_after"),
                    reset_oscillator_phase=reset_oscillator_phase,
                    triggers=trigger_output,
                    handle=handle,
                    user_register=user_register,
                    state=state,
                    local=local,
                ),
            )

            if parent_instance is not None:
                self.add_section_child(parent_instance, instance_id)

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
                            if pulse_offset_param is not None:
                                pulse_offset = self._all_parameters[pulse_offset_param]
                            (
                                pulse_amplitude,
                                pulse_amplitude_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "amplitude", (int, float, complex)
                            )
                            if pulse_amplitude_param is not None:
                                pulse_amplitude = self._all_parameters[
                                    pulse_amplitude_param
                                ]
                            (
                                pulse_increment,
                                pulse_increment_oscillator_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "increment_oscillator_phase", (int, float)
                            )
                            if pulse_increment_oscillator_phase_param is not None:
                                pulse_increment = self._all_parameters[
                                    pulse_increment_oscillator_phase_param
                                ]
                            (
                                pulse_set_oscillator_phase,
                                pulse_set_oscillator_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "set_oscillator_phase", (int, float)
                            )
                            if pulse_set_oscillator_phase_param is not None:
                                pulse_set_oscillator_phase = self._all_parameters[
                                    pulse_set_oscillator_phase_param
                                ]
                            (
                                pulse_phase,
                                pulse_phase_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "phase", (int, float)
                            )
                            if pulse_phase_param is not None:
                                pulse_phase = self._all_parameters[pulse_phase_param]
                            (
                                resulting_pulse_instance_length,
                                resulting_pulse_instance_length_param,
                            ) = find_value_or_parameter_dict(
                                pulse_ref, "length", (int, float)
                            )
                            if resulting_pulse_instance_length_param is not None:
                                resulting_pulse_instance_length = self._all_parameters[
                                    resulting_pulse_instance_length_param
                                ]

                            precompensation_clear = pulse_ref.get(
                                "precompensation_clear", False
                            )

                            pulse_id = None
                            if pulse_ref.get("pulse") is not None:
                                pulse_id = pulse_ref["pulse"]["$ref"]

                            acquire_params = None
                            signal_type = self._signals[signal_id].type.value
                            if signal_type == "integration":
                                acquire_params = AcquireInfo(
                                    handle=pulse_ref.get("readout_handle"),
                                    acquisition_type=getattr(
                                        self.acquisition_type, "value", None
                                    ),
                                )

                            pulse_parameters = copy.deepcopy(
                                pulse_ref.get("pulse_pulse_parameters")
                            )
                            operation_pulse_parameters = copy.deepcopy(
                                pulse_ref.get("play_pulse_parameters")
                            )

                            pulse_group = pulse_ref.get("pulse_group")
                            markers = []
                            for k, v in pulse_ref.get("markers", {}).items():
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
                                pulse=self._pulses[pulse_id]
                                if pulse_id is not None
                                else None,
                                signal=self._signals[signal_id],
                                offset=pulse_offset,
                                amplitude=pulse_amplitude,
                                length=resulting_pulse_instance_length,
                                acquire_params=acquire_params,
                                phase=pulse_phase,
                                increment_oscillator_phase=pulse_increment,
                                set_oscillator_phase=pulse_set_oscillator_phase,
                                play_pulse_parameters=operation_pulse_parameters,
                                pulse_pulse_parameters=pulse_parameters,
                                precompensation_clear=precompensation_clear,
                                markers=markers,
                                pulse_group=pulse_group,
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
